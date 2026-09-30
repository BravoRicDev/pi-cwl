/**
 * A live span must NOT switch off the deterministic eviction.
 *
 * THE DEFECT, MEASURED LIVE. In `index.ts` (~2419-2459), inside
 * `if (st.spans.length > 0)`, when at least one span applies
 * (`applied.applied > 0`) the hook renews the range address and does
 * `return finish(applied.kept)`. Everything below — the safety
 * floor, `reasoningFallback` (~2493) and `runEvictionPass` (~2524) — becomes
 * UNREACHABLE. And `applied.applied > 0` is true on EVERY turn as long as one span
 * resolves, because the span must be re-applied every time: so ONE SPAN IN THE STATE
 * SWITCHES OFF EVICTION AND FALLBACK FOR THE REST OF THE SESSION.
 *
 * The evidence was in the log of a real session (after the 2026-09-30 reload):
 *   `SPANS re-applied: 3, nothing new to count`
 *   `RANGE none | 178 msgs, 133852t vs trigger 68000t, 3 span(s)`
 * on every turn, with ZERO `EVICTION` lines, ZERO `CONTEXT ...`, ZERO `no safe
 * candidate`, ZERO `FALLBACK reasoning-strip`, while `cwl_status` declared
 * `Episodes: 2 | active: 0 | with evictable content: 1`. That is: 133k tokens
 * against a 68k trigger, one evictable episode, and the extension did
 * NOTHING — it neither evicted nor reduced, and it no longer even has an interval to ask
 * the agent for (`RANGE none`).
 *
 * The comment above that `return` proves the return was KNOWN: it had been
 * fixed for another reason (renewing the address, commit a7a8da0),
 * because leaving it before the renewal the agent could not compress a
 * second time. The consequence — eviction and fallback unreachable — had not
 * been seen. The test pins it down: a live span can save tokens, but it cannot
 * be the LAST thing the extension knows how to do.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const onlyRemoval = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(config) {
  const sandbox = makeSandbox({ name: `span-vs-eviction-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => {
  const p = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
};

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const text = (m) => JSON.stringify(m ?? {});
const contains = (out, marker) => out.some((m) => text(m).includes(marker));

/** Conversation with eligible endpoints: that is what makes an interval compressible. */
const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `prompt ${i} content. ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answers ${i} content is ` + 'A'.repeat(200) });
  }
  return out;
};

const assistant = (marker) => ({ role: 'assistant', content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }] });
const smallAssistant = (marker) => ({ role: 'assistant', content: [{ type: 'text', text: marker }] });

/**
 * The message that CARRIES the tool call.
 *
 * Why it is needed. The anchors of an episode are the TWO `toolResult`s of `delimiter`,
 * and the repair of the span pairs (`PAIR REPAIR`) DELETES the toolResults
 * left without their call. A test that invokes `delimiter` directly and
 * puts only the `toolResult`s in its own array builds ORPHAN anchors because
 * of the fixture, not of the code (in a real session the pair exists), and the
 * diagnosis ends up accusing the span of something the test did. The
 * shape of the block is the one used by `tests/compress-range.test.mjs`.
 */
const call = (id) => ({
  role: 'assistant',
  content: [{ type: 'toolCall', id, name: 'delimiter', arguments: { action: 'end', name: 'after-span-1' } }],
});

/** Creates a real span, with the same tool the agent uses in production. */
async function createSpan(tools, hooks, ctx, messages) {
  await hook(hooks, ctx, messages);
  const out = await tools.get('cwl_compress_range').execute('t', { summary: 'SUMMARY of the first turn' }, undefined, undefined, ctx);
  assert.equal(out.details.ok, true, `the span was not created (without a span the test proves nothing): ${JSON.stringify(out.details)}`);
  return out;
}

test('with a live span that re-applies, the evictable episode is still evicted', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval());
  try {
    // 1. A compressible conversation: it is needed to create the span.
    const base = conversation();
    await createSpan(tools, hooks, ctx, base);

    // 2. A closed episode, with large content, AFTER the span: it is evictable.
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'after-span-1', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'after-span-1', description: 'takeaway' }, undefined, undefined, ctx);
    const messages = [
      ...base,
      { role: 'user', content: 'open' },
      call('call-s'),
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      assistant('INSIDE-1'),
      assistant('INSIDE-2'),
      call('call-e'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];

    const out = await hook(hooks, ctx, messages);
    const log = logOf(sandbox);
    // Non-vacuity: the span MUST have re-applied on this turn, otherwise
    // the test would pass even without the defect it wants to demonstrate.
    assert.match(log, /SPANS (re-applied|applied): \d+/, 'the span did not re-apply: the test proves nothing');

    assert.ok(
      !contains(out, 'INSIDE-1') || !contains(out, 'INSIDE-2'),
      'the context is above the trigger with an evictable episode, the span re-applied, and the content of ' +
      'the episode is still there: the span branch returned BEFORE `runEvictionPass` and `reasoningFallback`, ' +
      'so a single span in the state switches off eviction for the rest of the session',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('if the span is enough to get back within the budget, the episode is NOT touched', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval({ tokenBudget: 1500 }));
  try {
    const base = conversation();
    await createSpan(tools, hooks, ctx, base);

    // Small episode: after the span the context gets back within the trigger, so there is
    // nothing to evict and the eviction must not touch anything. It is the
    // insurance against an overly aggressive fix ("always evict").
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'minimal', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'minimal', description: 'takeaway' }, undefined, undefined, ctx);
    const messages = [
      ...base,
      { role: 'user', content: 'open' },
      call('call-s'),
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      smallAssistant('SMALL-1'),
      call('call-e'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'current' },
    ];

    const out = await hook(hooks, ctx, messages);
    assert.ok(contains(out, 'SMALL-1'), 'the episode was evicted while the context was already back within the trigger: no reason to touch it');
    // And the compressed list MUST be the one returned: without the early
    // return in the span branch, the flow falls into the tail, which when it is below
    // the trigger answers `undefined` (no change) and the provider receives the
    // NON-compressed history. The span saving would be lost for that turn.
    assert.ok(contains(out, 'SUMMARY'), 'the applied span was not returned: the compression was lost');
  } finally { home.restore(); sandbox.cleanup(); }
});
