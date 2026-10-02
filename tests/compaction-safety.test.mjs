/**
 * Historical reasoning stays intact until explicit block compression.
 * No global fallback may rewrite messages outside annotated episodes,
 * even above budget. The range tool must remain available independently.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const baseConfig = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: true, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(config) {
  const sandbox = makeSandbox({ name: `safety-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const thinking = (tag) => ({
  id: `a-${tag}`,
  role: 'assistant',
  content: [{ type: 'thinking', thinking: `${tag}:` + 'T'.repeat(4000) }],
});
const hasThinking = (m) => Array.isArray(m?.content) && m.content.some((b) => b?.type === 'thinking');

test('with an episode present, outside reasoning remains unchanged without a global fallback', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(baseConfig());
  try {
    // An episode exists: it is exactly the condition that used to switch the net off.
    await tools.get('delimiter').execute('call-e', { name: 'ep', type: 'expl', description: 'n' }, undefined, undefined, ctx);

    const messages = [
      { role: 'user', content: 'opening' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      // These sit OUTSIDE any episode.
      thinking('outer-1'), thinking('outer-2'), thinking('outer-3'),
      { role: 'user', content: 'inquiry' },
    ];
    const beforeCount = messages.filter(hasThinking).length;
    assert.equal(beforeCount, 3);

    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    assert.equal(out.filter(hasThinking).length, 3,
      'a global fallback rewrote reasoning outside the episode');
    assert.deepEqual(out.filter(hasThinking), messages.filter(hasThinking));
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the safety window: the last N user turns are not touched', async () => {
  const N = 10;
  const { sandbox, home, hooks, ctx } = await boot(baseConfig({ protectedTurns: N }));
  try {
    // 12 turns: user + assistant(thinking) each.
    const messages = [];
    for (let i = 1; i <= 12; i++) {
      messages.push({ role: 'user', content: `round ${i}` });
      messages.push(thinking(`t${i}`));
    }
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;

    const survivors = out.filter(hasThinking).map((m) => m.content[0].thinking.slice(0, m.content[0].thinking.indexOf(':')));
    // Neither old nor protected turns are rewritten by a global fallback.
    assert.ok(survivors.includes('t1'), 'old reasoning must remain intact until block compression');
    for (let i = 12 - N + 1; i <= 12; i++) {
      assert.ok(survivors.includes(`t${i}`), `turn ${i} is inside the safety window and had not to be touched`);
    }
    assert.equal(out.filter((m) => m.role === 'user').length, 12, 'a user turn was removed');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('without a window, reasoning remains intact until explicit block compression', async () => {
  const { sandbox, home, hooks, ctx } = await boot(baseConfig({ protectedTurns: 0 }));
  try {
    const messages = [];
    for (let i = 1; i <= 12; i++) {
      messages.push({ role: 'user', content: `round ${i}` });
      messages.push(thinking(`t${i}`));
    }
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    assert.equal(out.filter(hasThinking).length, 12);
    assert.deepEqual(out, messages);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a conversation shorter than the window releases ONLY the oldest part', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(baseConfig({ protectedTurns: 10 }));
  try {
    // 3 turns, window of 10: the window would cover EVERYTHING. But a window that
    // covers everything is no longer a window: it is the reason the context
    // grows without bound, because all three compaction routes die
    // together. MEASURED in real sessions: 469k tokens against a threshold of 68k,
    // `cwl_compress_range` answering "nothing left to compress", and the
    // gate looping on an impossible request.
    //
    // Rule: in the degenerate case the OLDEST part is released, the
    // recent one stays intact.
    const messages = [
      { role: 'user', content: '1st' }, thinking('t1'),
      { role: 'user', content: '2nd' }, thinking('t2'),
      { role: 'user', content: '3rd' }, thinking('t3'),
    ];
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    const left = out.filter(hasThinking).length;
    assert.equal(left, 3, 'the hook must not rewrite historical thinking');
    const compressed = await tools.get('cwl_compress_range').execute('compress-old', {
      summary: 'Historical work summary', micro: 'Earlier work, available in full through the leaf.',
    }, undefined, undefined, ctx);
    assert.equal(compressed.details.ok, true, 'block compression must remain available without fallback');
    const applied = await hooks.get('context')({ messages }, ctx);
    assert.ok((applied?.messages ?? messages).filter(hasThinking).length < 3,
      'the explicit leaf must replace older reasoning');
    assert.equal(out.filter((m) => m.role === 'user').length, 3, 'the user turns are inviolable');
  } finally { home.restore(); sandbox.cleanup(); }
});
