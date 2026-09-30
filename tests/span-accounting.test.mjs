/**
 * The saving a span DECLARES must be what the context has lost.
 *
 * THE DEFECT. In `applySpans` the gain of a span was computed over the whole
 * range:
 *
 *   for (let i = from; i <= to; i++) { replaced.add(i); original += estimateMessageTokens(messages[i]); }
 *   const gain = Math.max(0, original - estimateTokens(sp.summary));
 *
 * but the range is NOT removed entirely: further down, when the list is
 * rebuilt, the `user`, `system`, `developer` and `custom` messages that fall
 * inside the span are KEPT (`if (role === 'user' || ...) { kept.push(m); }`).
 * Those tokens stay in the context and keep costing, but the gain counted them as
 * saved. MEASURED in a sandbox with a conversation of 12
 * messages (6 `user` + 6 `assistant`, ~756 tokens): the log declared
 * `SPANS applied: 1 (new 1), saved 749t` while the six `user` messages were
 * still present in the returned list, so the context had dropped by ~338t.
 * The span declared more than double what it had freed.
 *
 * It is the same family as the defect closed in the eviction accounting
 * (`saved` was the plan's estimate instead of `B-A`): a number claiming more than
 * the work done. The consequence is not cosmetic: `cwl_status` adds up those
 * savings, so the extension believes it has freed space it still has, and
 * stops looking for levers exactly when the context is still above the trigger
 * (live: ~142,000 tokens measured against a trigger of 68,000).
 *
 * The same wrong number ended up in the message injected in place of the
 * compressed messages, which declares `(~N token risparmiati)` by passing the
 * size of the RANGE.
 *
 * The remedy is the one of the eviction: count what has REALLY been removed,
 * with a single definition of the surviving roles used both by the count and
 * by the rebuild of the list, so that the two can no longer diverge.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Deliberately low budget: the trigger (300) must stay below the context
 * even AFTER the span, because it is the line the extension writes in that case
 * (`SPANS applied (Nt) still above trigger Mt`) that gives us the after measure.
 */
const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `span-accounting-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

/** 6 `user` + 6 `assistant`: half of the range survives the span (the user turns). */
const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

test('the saving declared by a span is what the context has really lost', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversation();
    // Turn 1: the extension takes the measurements and stores the address to compress.
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 of the first turns' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the span was not created (without a span the test proves nothing): ${JSON.stringify(comp.details)}`);

    // Turn 2: the span is applied for the first time.
    const out = await hook(hooks, ctx, base);
    const log = logOf(sandbox);

    const before = /RANGE \S+ \(~\d+t\) \| 12 msgs, (\d+)t vs trigger/.exec(log);
    const declared = /SPANS applied: \d+ \(new \d+\), saved (\d+)t/.exec(log);
    const after = /SPANS applied \((\d+)t\) still above trigger/.exec(log);
    assert.ok(
      before && declared && after,
      'all three lines are needed (measure before, declared saving, measure after): ' +
      `before=${!!before} declared=${!!declared} after=${!!after}`,
    );

    // NON-VACUITY: the `user` turns inside the span really survive, so the
    // declared figure has a chance to overshoot. If one day the span removed them, this
    // test must be rethought instead of staying green by accident.
    assert.ok(JSON.stringify(out).includes('turn 1 content'),
      'the `user` turns inside the span turn out to be removed: the test\'s premise no longer holds');

    assert.equal(
      Number(declared[1]),
      Number(before[1]) - Number(after[1]),
      `the span declares it saved ${declared[1]}t but the context dropped from ${before[1]}t to ${after[1]}t, ` +
      `that is ${Number(before[1]) - Number(after[1])}t: the user turns inside the span stay in the context and keep costing`,
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the injected message no longer declares more tokens than the context has lost', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversation();
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 of the first turns' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the span was not created: ${JSON.stringify(comp.details)}`);

    const out = await hook(hooks, ctx, base);
    const log = logOf(sandbox);
    const before = /RANGE \S+ \(~\d+t\) \| 12 msgs, (\d+)t vs trigger/.exec(log);
    const after = /SPANS applied \((\d+)t\) still above trigger/.exec(log);
    assert.ok(before && after, 'measurements missing: the test proves nothing');
    const lost = Number(before[1]) - Number(after[1]);

    const injected = out.find((m) => m?.customType === 'cwl-compressed');
    assert.ok(injected, 'the compressed summary was not injected');
    const claim = /~(\d+) token risparmiati/.exec(JSON.stringify(injected));
    assert.ok(claim, 'the injected message does not declare the saved tokens');
    assert.ok(
      Number(claim[1]) <= lost,
      `the injected message declares ~${claim[1]} token risparmiati but the context has lost ${lost}: ` +
      'the number was the size of the RANGE, not the saving',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});
