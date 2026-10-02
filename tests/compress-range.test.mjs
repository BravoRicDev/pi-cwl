/**
 * `cwl_compress_range`: the extension computes the address, the model writes the text.
 *
 * Why it exists. The budget gate asked to compact with
 * `cwl_compress(startHash=..., endHash=...)`. In a REAL session that request
 * was not executable: the hashes live in the state, they are opaque and the agent
 * has no way to know which hash corresponds to which message. Measured result:
 * gate armed at turn 2, 3 attempts, zero compactions, context stuck at 370k
 * against a threshold of 68k.
 *
 * The split that works: the extension picks the addresses (it has them), the
 * model writes the summary (the only part that only the model can do).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const config = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 2,
  gate: false,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(cfg) {
  const sandbox = makeSandbox({ name: `range-${seq++}`, config: cfg });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Six user+assistant turns: the last 2 stay in the safety window. */
const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const call = (tools, ctx, summary) =>
  tools.get('cwl_compress_range').execute('t', { summary }, undefined, undefined, ctx);

test('cwl_compress_range compresses the oldest range, without hashes', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, 'summary of the first turns: goal, decisions, paths');
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);
    assert.ok(out.details.tokens > 0, 'the compressed range must have tokens');
    // The leaf id belongs in the ANSWER: the marker in the context carries it too, but a turn
    // that compresses twice cannot read a marker it has not seen yet, and cannot group by name.
    const leafId = String(out.details.id);
    assert.match(leafId, /^sp-[0-9a-f]{8}$/, `the answer must name the leaf it just created: ${JSON.stringify(out.details)}`);
    assert.ok(
      out.content.map((c) => c.text).join('\n').includes(leafId),
      'the leaf id is in the details but not in the text the model reads',
    );
    // The address has been consumed: a second shot cannot re-compress the same one.
    const again = await call(tools, ctx, 'second attempt');
    assert.equal(again.details.ok, false);
    assert.equal(again.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the safety window is not touched: the range stops earlier', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 2 }));
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'summary');
    assert.equal(out.details.ok, true);

    // Direct check: the range hashes must NOT include the messages
    // of the last 2 turns (the last 4 messages of the list).
    const { createHash } = await import('node:crypto');
    const hashOf = (s) => createHash('sha256').update(s).digest('hex').slice(0, 12);
    const protectedHashes = messages.slice(-4).map((m) => hashOf(m.content));
    assert.ok(!protectedHashes.includes(out.details.startHash ?? ''), 'start inside the window');
    assert.ok(!protectedHashes.includes(out.details.endHash ?? ''), 'end inside the window');
    // And it must instead include the first message (outside the window).
    assert.equal(out.content[0].text.includes('Compressed'), true);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('with nothing to compress it refuses instead of inventing a range', async () => {
  // Two messages only: before the window there is not enough material to
  // form a range (a pair is needed), so there is nothing to take.
  // The case "window so wide it covers everything" is NO longer a refusal: the
  // oldest part is released on purpose, otherwise the extension could
  // never close the context (measured: 469k tokens, 68k threshold, no
  // compaction possible).
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 99 }));
  try {
    const minimal = [
      { role: 'user', content: 'question ' + 'U'.repeat(200) },
      { role: 'assistant', content: 'answer ' + 'A'.repeat(200) },
    ];
    await hooks.get('context')({ messages: minimal }, ctx);
    const out = await call(tools, ctx, 'summary');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('an empty summary is rejected: it is the only part the model must write', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, '   ');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'missing-summary');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('an already-compressed range is not proposed again', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const first = await call(tools, ctx, 'first summary');
    assert.equal(first.details.ok, true);
    // The next hook recomputes the range ON WHAT REMAINS.
    await hooks.get('context')({ messages }, ctx);
    const status = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
    const text = status.content.map((c) => c.text).join('\n');
    // Either nothing remains, or a range different from the first one remains.
    const hasRange = /Compressible range/.test(text);
    if (hasRange) {
      assert.ok(!text.includes(`${first.details.startHash}..${first.details.endHash}`),
        'the already-compressed range is proposed again');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the saving is counted ONCE: re-applying the span does not inflate the total', async () => {
  // Pi rebuilds the list from the transcript at each turn, so it hands back the
  // ORIGINALS: the span must be re-applied, and it is right that it is so.
  // Re-applying is not saving again — the context stays the compressed
  // one, it is not compressed twice. Counting the saving at each
  // re-application produces a number that grows by itself, turn after turn, and
  // that number is the only proof the operator has that the thing works.
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  const savedTokens = async () => {
    const s = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
    const text = s.content.map((c) => c.text).join('\n');
    const m = /tokens saved:\s*([\d.]+)/.exec(text);
    return m ? Number(m[1].replace(/\./g, '')) : -1;
  };
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'summary of the first turns: goal, decisions, paths');
    assert.equal(out.details.ok, true);
    // The counting happens when the span is APPLIED, that is at the next
    // turn: the tool writes the address, the hook applies it.
    await hooks.get('context')({ messages }, ctx);
    const afterCompression = await savedTokens();
    assert.ok(afterCompression > 0, 'the compression must have saved something');

    // Three turns in which exactly the same list is handed back: no new
    // compression, no new eviction. The total must not move.
    for (let i = 0; i < 3; i++) await hooks.get('context')({ messages }, ctx);
    const afterThreeTurns = await savedTokens();
    assert.equal(afterThreeTurns, afterCompression,
      `the saving grew without new compressions: ${afterCompression} -> ${afterThreeTurns}`);
  } finally { home.restore(); sandbox.cleanup(); }
});

/** Broken toolCall/toolResult pairs: they are exactly what the provider rejects. */
const orphans = (messages) => {
  const calls = new Set();
  const results = new Set();
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const b of m.content) if (b && b.type === 'toolCall' && b.id) calls.add(b.id);
    }
    if (typeof m.toolCallId === 'string') results.add(m.toolCallId);
  }
  return {
    callsWithoutResult: [...calls].filter((id) => !results.has(id)),
    resultsWithoutCall: [...results].filter((id) => !calls.has(id)),
  };
};

/**
 * Regression: the compression must not split a toolCall/toolResult pair.
 *
 * Measured on a REAL session. The compressed range ended on an assistant
 * that carried a toolCall (index 1296) while its toolResult, 6396 char,
 * stayed outside (index 1297). In the context a `tool_result` survived without
 * its `tool_use`, the provider answered `400 status code (no body)` and the
 * session got stuck: no useful error message, no way to
 * resume except by hand.
 *
 * The deterministic eviction path ALREADY knew this invariant and
 * respected it (H1, `droppedToolCallIds`: "the assistant message that carries the
 * matching toolCall must not keep it, or the conversation has a dangling...").
 * The span path had no guard at all.
 *
 * Why the case shows up only NOW: in the normal case `protectedFromIndex`
 * returns `index_user + 1`, so the range always ends on a user message.
 * Only in the DEGENERATE case (fewer user turns than the window) the floor
 * is an arbitrary share of the list, and then it can fall right after an
 * assistant. It is the same condition as the fix "the safety window covered
 * the WHOLE list": 469k tokens against a 68k threshold.
 */
test('the compression leaves no orphan toolResult', async () => {
  // protectedTurns 4 with only 2 user turns: degenerate case, the floor
  // falls halfway down the list — exactly where it fell in the real session.
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 4 }));
  try {
    const text = (t) => `${t} ` + 'X'.repeat(240);
    const exchange = (n) => ([
      { role: 'assistant', content: [{ type: 'text', text: text(`thinking ${n}`) }, { type: 'toolCall', id: `tc${n}`, name: 'bash', arguments: { command: 'ls' } }] },
      { role: 'toolResult', toolCallId: `tc${n}`, content: [{ type: 'text', text: text(`output ${n}`) }] },
    ]);
    const messages = [
      { role: 'user', content: text('turn 1') },
      ...exchange(1),
      ...exchange(2),
      ...exchange(3),
      { role: 'user', content: text('turn 2') },
    ];

    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'summary of the turns with tools');
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);

    // The hook is the only point where the list is rewritten: it is what goes to the provider.
    const res = await hooks.get('context')({ messages }, ctx);
    const kept = res.messages;
    const { callsWithoutResult, resultsWithoutCall } = orphans(kept);
    assert.deepEqual(resultsWithoutCall, [],
      `tool_result without its tool_use: the provider answers 400. Orphans: ${resultsWithoutCall.join(', ')}`);
    assert.deepEqual(callsWithoutResult, [],
      `tool_use without its result: the provider answers 400. Orphans: ${callsWithoutResult.join(', ')}`);
    // The safety window stays untouchable: extending the range to the
    // toolResult must not become "eat everything to the bottom".
    assert.ok(kept.some((m) => m.role === 'user' && String(m.content).includes('turn 2')),
      'the protected user turn must survive');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('bulk results are not repaired by position: the guarantee is by id', async () => {
  // Layout MEASURED in a real session (2026-09-23T07-11-59, lines 1054-1061):
  // Pi writes the FOLLOWING assistant turn before the block of results,
  // so an assistant can sit BETWEEN a toolCall and its result.
  //   1054 assistant  toolCall x5   (call_00_kzf4is ...)
  //   1055 assistant  toolCall x1   <- in between, without results
  //   1056 toolResult of 1054       <- arrives after
  // Here the assistant B sits between the call of A and the result of A. The range
  // ends on A, so the call of A disappears and its result stays orphan:
  // a rule by position CANNOT see it (no toolResult is contiguous
  // with the last message of the range). Staying orphan means that the provider
  // rejects the whole request: 400 from the gateway, or
  // "No tool call found for function call output with call_id ..." from Codex.
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 4, debug: true }));
  try {
    const text = (t) => `${t} ` + 'X'.repeat(240);
    const messages = [
      { role: 'user', content: text('turn 1') },
      // Last valid endpoint below the floor: the range ends here.
      { role: 'assistant', content: [{ type: 'text', text: text('A thinks') }, { type: 'toolCall', id: 'ca', name: 'bash', arguments: { command: 'ls' } }] },
      // Index >= floor: it stays outside, with its pair intact.
      { role: 'assistant', content: [{ type: 'text', text: text('B thinks') }, { type: 'toolCall', id: 'cb', name: 'bash', arguments: { command: 'ls' } }] },
      // The result of A arrives AFTER the assistant B: its call is inside the
      // range and disappears, so this result is orphan and must be discarded.
      { role: 'toolResult', toolCallId: 'ca', content: [{ type: 'text', text: text('output from A') }] },
      { role: 'toolResult', toolCallId: 'cb', content: [{ type: 'text', text: text('output from B') }] },
    ];

    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'summary of exchange A');
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);

    const res = await hooks.get('context')({ messages }, ctx);
    const kept = res.messages;
    // Without applying the span there would be nothing to repair and the test
    // would pass EMPTY: it is the mistake already made once.
    assert.ok(kept.some((m) => m.customType === 'cwl-compressed'),
      'the span must have been applied, otherwise the test proves nothing');

    const { callsWithoutResult, resultsWithoutCall } = orphans(kept);
    assert.deepEqual(resultsWithoutCall, [],
      `tool_result without its tool_use: the provider answers 400. Orphans: ${resultsWithoutCall.join(', ')}`);
    assert.deepEqual(callsWithoutResult, [],
      `tool_use without its result: the provider answers 400. Orphans: ${callsWithoutResult.join(', ')}`);
    // Opposite mutation ("repair by discarding everything"): the pair of B has nothing
    // to do with the range and must survive ENTIRE, call and result.
    assert.ok(kept.some((m) => m.role === 'toolResult' && m.toolCallId === 'cb'),
      'the result of B, extraneous to the range, must not be discarded');
    assert.ok(kept.some((m) => Array.isArray(m.content)
      && m.content.some((b) => b.type === 'toolCall' && b.id === 'cb')),
      'the toolCall of B must stay: its result is alive');

    // The repair DISCARDS a message: if it did not say so it would be a silent
    // change, and a silent change is the way a bug hides.
    // The log is the only trace, so it must be verified instead of promised.
    const fs = await import('node:fs');
    const log = fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');
    assert.match(log, /PAIR REPAIR: dropped 1 orphan tool result/,
      `the log must say it discarded the orphan result. Log:\n${log}`);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('two messages with the SAME text do not collapse onto a single address', async () => {
  // MEASURED in a real session: 303 assistant messages with the text "*", 51
  // with "🌙", 40 with the same 100-character sentence. With only the text
  // hash they all collapse onto ONE address, and the map keeps the LAST occurrence:
  // the span created on the first resolved onto the last, that is onto a range
  // DIFFERENT from the one requested — even beyond the safety window.
  // The transcript carries a timestamp on each message (epoch ms for the
  // assistant) and it is that which distinguishes two identical texts.
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 4 }));
  try {
    const text = (t) => `${t} ` + 'X'.repeat(240);
    const messages = [
      { role: 'user', content: text('turn 1'), timestamp: 1000 },
      // Last valid endpoint below the floor: the range ends here.
      { role: 'assistant', content: 'ok', timestamp: 2000 },
      // A non-endpoint message, to make the floor fall after the first "ok".
      { role: 'custom', customType: 'other', content: 'injected recap', timestamp: 2500 },
      // SAME TEXT as the first, but beyond the window: here the ambiguous address
      // made the span reach this far, swallowing it.
      { role: 'assistant', content: 'ok', timestamp: 9000 },
      { role: 'user', content: text('turn 2'), timestamp: 9500 },
      { role: 'assistant', content: text('recent answer'), timestamp: 9600 },
    ];

    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'summary of the first turn');
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);

    const res = await hooks.get('context')({ messages }, ctx);
    const kept = res.messages;
    assert.ok(kept.some((m) => m.customType === 'cwl-compressed'),
      'the span must have been applied, otherwise the test proves nothing');
    const hasFirstOk = (l) => l.some((m) => m.role === 'assistant' && m.content === 'ok' && m.timestamp === 2000);
    const hasSecondOk = (l) => l.some((m) => m.role === 'assistant' && m.content === 'ok' && m.timestamp === 9000);
    assert.ok(!hasFirstOk(kept), 'the first "ok" is inside the range: the summary replaces it');
    assert.ok(hasSecondOk(kept),
      'the second "ok" is BEYOND the window: with the ambiguous address the span reached that far and erased it');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('an episode whose closing anchor left the context is STATED, not assumed evacuated', async () => {
  // MEASURED in a real session: all 4 episodes were at level='removed',
  // but their delimiter anchors had been taken away by the native
  // compaction. episodeRanges skips them (right: without anchors it does not know what to touch),
  // however recoverable() excludes the 'removed' episodes, so the extension
  // believed it had evacuated them and declared itself fine. The real defect was not
  // the content: it was the SILENCE.
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const delim = (id, params) => tools.get('delimiter').execute(id, params, undefined, undefined, ctx);
    const closeResult = await delim('T2', { name: 'ep-anchor', type: 'expl', description: 'done' });
    assert.equal(closeResult.details.ok, true, `closing rejected: ${JSON.stringify(closeResult.details)}`);

    const body = 'X'.repeat(200);
    const complete = [
      { role: 'user', content: `before ${body}` },
      { role: 'assistant', content: `work ${body}` },
      { role: 'toolResult', toolCallId: 'T1', content: [{ type: 'text', text: 'episode start' }] },
      { role: 'assistant', content: `inside ${body}` },
      { role: 'toolResult', toolCallId: 'T2', content: [{ type: 'text', text: 'episode end' }] },
      { role: 'user', content: `after ${body}` },
      { role: 'assistant', content: `last ${body}` },
    ];

    await hooks.get('context')({ messages: complete }, ctx);
    const withAnchors = await tools.get('cwl_status').execute('t', {}, undefined, undefined, ctx);
    assert.equal(withAnchors.details.unlocatable, 0,
      'with the anchors in the context the episode is locatable: ' + JSON.stringify(withAnchors.details));

    // The native compaction takes away the CLOSING anchor: with the implicit open the
    // start needs no anchor, so the closing is the only endpoint and the episode is no
    // longer locatable. It must be STATED.
    const withoutAnchors = complete.filter((m) => m.toolCallId !== 'T2');
    await hooks.get('context')({ messages: withoutAnchors }, ctx);
    const without = await tools.get('cwl_status').execute('t', {}, undefined, undefined, ctx);
    assert.equal(without.details.unlocatable, 1,
      'an unlocatable episode must be counted and stated: ' + JSON.stringify(without.details));
  } finally { home.restore(); sandbox.cleanup(); }
});

test('after the first compression the address is renewed: you can compress again', async () => {
  // The span branch exited with `return` BEFORE the point that computes the
  // next range (~2290). But `cwl_compress_range` does not recompute it —
  // it does not have the list of messages, it reads `st.rangeStartHash` — so as long as
  // a span resolved, the address was NEVER renewed and the agent could no
  // longer compress, while the conversation kept growing.
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const turn = (i) => ([
      { role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) },
      { role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) },
    ]);
    const list = [];
    for (let i = 1; i <= 6; i++) list.push(...turn(i));

    await hooks.get('context')({ messages: list }, ctx);
    const firstCall = await call(tools, ctx, 'summary of the first half');
    assert.equal(firstCall.details.ok, true,
      `the first compression must pass: ${JSON.stringify(firstCall.details)}`);

    // The conversation grows: new turns enter the compressible zone.
    for (let i = 7; i <= 10; i++) list.push(...turn(i));

    await hooks.get('context')({ messages: list }, ctx);
    const secondCall = await call(tools, ctx, 'summary of the second half');
    assert.equal(secondCall.details.ok, true,
      'after the first compression the address must be renewed, or the agent can no longer compress: '
      + JSON.stringify(secondCall.details));
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a span whose endpoints left the context is pruned, and it shows', async () => {
  // A non-applicable span does no harm to the context, but its existence is a
  // silence: its summary (thousands of characters) is re-saved into the
  // state at every turn. MEASURED: in the state file of a real session, 4 spans
  // carried 38,459 characters of summaries on 51,880 bytes of file.
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const turn = (i) => ([
      { role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) },
      { role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) },
    ]);
    const list = [];
    for (let i = 1; i <= 6; i++) list.push(...turn(i));

    await hooks.get('context')({ messages: list }, ctx);
    const out = await call(tools, ctx, 'summary of the first half');
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);
    assert.equal(out.details.spans, 1, 'the span just created must exist');

    const live = await tools.get('cwl_status').execute('t', {}, undefined, undefined, ctx);
    assert.equal(live.details.spans, 1, 'a span with its endpoints in the context must NOT be pruned');

    // A native compaction replaces that history: the endpoints of the span
    // are no longer in the list, and they will not come back.
    const afterCompaction = list.slice(-2);
    await hooks.get('context')({ messages: afterCompaction }, ctx);

    const stat = await tools.get('cwl_status').execute('t', {}, undefined, undefined, ctx);
    assert.equal(stat.details.spans, 0,
      'the span whose endpoints left the context must be pruned: ' + JSON.stringify(stat.details));
  } finally { home.restore(); sandbox.cleanup(); }
});
