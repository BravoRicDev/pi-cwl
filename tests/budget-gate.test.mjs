/**
 * Level B: the gate that asks the AGENT to compact.
 *
 * Why it exists. Deterministic eviction is blind: it cuts by age, not by
 * meaning. The gate is the only channel that can get a semantic decision out of
 * the model.
 *
 * Key difference from anti-amnesia: there the gate closes when the model
 * ECHOES a `[CARD OK]` token; here it closes by EFFECT, because the thing we
 * want (fewer tokens) is directly measurable. An invented confirmation is worth
 * nothing: only the measured context counts.
 *
 * The gate is a PREFERENCE, never a guarantee: if the model ignores it, the
 * Level A parachute is what keeps holding the context.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// No active level: so NOTHING gets compacted deterministically and the context
// stays above budget, which is the condition in which the gate must act.
const overBudgetConfig = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  gate: true,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(config) {
  const sandbox = makeSandbox({ name: `gate-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Messages without thinking: nothing to compact, so it stays above budget. */
const heavy = (n = 6) =>
  Array.from({ length: n }, (_, i) => ({ role: 'user', content: `message ${i} ${'P'.repeat(400)}` }));

const gateOf = (res) => (res?.messages ?? []).filter((m) => m.customType === 'cwl-budget-gate');
/** A full round: context hook + end of turn. */
async function round(hooks, ctx, messages) {
  const res = await hooks.get('context')({ messages }, ctx);
  await hooks.get('turn_end')({}, ctx);
  return res;
}

test('the gate injects the compact request when the budget does not drop', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    // Turns 1 and 2: above budget but the gate is not armed yet.
    const r1 = await round(hooks, ctx, messages);
    assert.equal(gateOf(r1).length, 0, 'the gate must not ask for anything on the first turn');
    const r2 = await round(hooks, ctx, messages);
    assert.equal(gateOf(r2).length, 0, 'the gate must not ask for anything on the second turn');

    // Turn 3: turn_end armed the gate, so the next hook raises it.
    const r3 = await round(hooks, ctx, messages);
    const demands = gateOf(r3);
    assert.equal(demands.length, 1, 'the gate should have injected the request');
    const text = String(demands[0].content);
    assert.match(text, /cwl_compress/, 'the request does not say what to do');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the request does not pile up: it is replaced, not stacked', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    // The hook's RESPONSE is chained into the NEXT one, as Pi does: it is the
    // only way to notice a pile-up. Always passing the same starting array, the
    // previous request would never enter the next round, and a missing filter
    // would stay invisible.
    let cur = heavy();
    for (let i = 0; i < 5; i++) {
      const res = await hooks.get('context')({ messages: cur }, ctx);
      cur = res?.messages ?? cur;
      assert.ok(gateOf(res).length <= 1,
        `round ${i}: the context holds ${gateOf(res).length} requests together`);
      await hooks.get('turn_end')({}, ctx);
    }
    assert.equal(gateOf({ messages: cur }).length, 1,
      'after arming the request must be there, and be a single one');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the gate closes by EFFECT: context under budget = no request', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const active = await round(hooks, ctx, messages);
    assert.equal(gateOf(active).length, 1, 'the gate should have been active');

    // The context drops below the threshold (trigger = 100 * 0.5 = 50 tokens).
    const light = [{ role: 'user', content: 'short' }];
    const under = await hooks.get('context')({ messages: light }, ctx);
    assert.equal(gateOf(under).length, 0, 'under budget the request must not appear');

    // And the gate is DISARMED: the next hook, again above budget, must not ask
    // for anything. Without the disarm it would stay armed and ask again
    // immediately, which is exactly what this test must catch.
    const after = await hooks.get('context')({ messages: messages }, ctx);
    assert.equal(gateOf(after).length, 0,
      'the gate stayed armed after being satisfied');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('after N attempts without an answer the gate gives up (with cooldown)', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    // You reach the active gate, then let the attempts go by.
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    assert.equal(gateOf(await round(hooks, ctx, messages)).length, 1, 'gate not active');

    // GATE_MAX_ATTEMPTS = 3: the limit is exceeded.
    for (let i = 0; i < 5; i++) await round(hooks, ctx, messages);
    const r = await round(hooks, ctx, messages);
    assert.equal(gateOf(r).length, 0,
      'after the maximum number of attempts the gate must fall silent instead of insisting forever');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('with gate=false no request is ever injected', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig({ gate: false }));
  try {
    const messages = heavy();
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0, 'gate=false must disable the channel entirely');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('with NOTHING to do the gate does not ask: it must never ask the impossible', async () => {
  // Reproduces a real session. Measured there: the gate asked to compact, but
  //  - no episode was open (all 4 closed)  -> option 1 unavailable
  //  - cwl_compress_range answered "nothing left to compress"
  //    -> option 2 unavailable
  // So it asked for something impossible every turn, burning context.
  //
  // Here: just TWO messages. Before the window there is not enough material
  // to form a range (a pair is needed), and no episode is open:
  // there is really nothing to do.
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    const messages = [
      { role: 'user', content: 'question ' + 'P'.repeat(400) },
      { role: 'assistant', content: 'reply ' + 'R'.repeat(400) },
    ];
    for (let i = 0; i < 8; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0,
        `turn ${i}: the gate asked to compact without having anything to propose`);
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('with an OPEN episode the gate asks, and proposes only the available option', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    // An open episode is the only action really available here.
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'work-in-progress', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
      { role: 'assistant', content: `reply ${'R'.repeat(400)}` },
    ];
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const r = await round(hooks, ctx, messages);
    const demands = gateOf(r);
    assert.equal(demands.length, 1, 'with an open episode the gate must ask');
    const text = String(demands[0].content);
    assert.match(text, /delimiter/, 'it must propose closing the episodes');
    assert.doesNotMatch(text, /cwl_compress_range/,
      'it must not propose an option that is not available (no compressible range)');
  } finally { home.restore(); sandbox.cleanup(); }
});

/** The anchor of an episode is the tool call that opened it, by call id. */
const anchor = (callId, text = 'R'.repeat(400)) => ({ role: 'toolResult', toolCallId: callId, content: text });

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

test('an episode that BEGINS at the end of the list is not worth closing, and the gate stays silent', async () => {
  // Measured shape of the impossible ask: the only open episode starts on the LAST
  // message, so its range would be `to <= from` and closing it frees nothing. Asking
  // for it costs a turn and dirties the context, and the agent cannot refuse it in a
  // useful way: it would close the episode and watch the context stay exactly as big.
  // Before this check the gate asked here — the condition was the static
  // `active().length > 0` — and the demand was unsatisfiable BY CONSTRUCTION.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10, debug: true }));
  try {
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'at-the-end', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
      anchor('call-s'),
    ];
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0,
        `turn ${i}: the gate asked to close an episode that saves nothing`);
    }
    // And it says WHY it stays silent: a withheld demand that leaves no trace is
    // the same defect one level down.
    assert.match(logOf(sandbox), /GATE withheld: the only open episode\(s\) begin at the end of the list/,
      'the withheld demand was not explained in the log');
    // A demand that was never DELIVERED is not a failure of the agent. Before this
    // check the attempt counter moved anyway — `turn_end` looked only at its own
    // static predicate — and after three turns the gate gave up with "unanswered for
    // 3 turns" on a request the agent had never seen.
    assert.doesNotMatch(logOf(sandbox), /GATE dropped/,
      'the gate gave up on a demand it never delivered');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the same episode, with material after its anchor, IS worth closing and the gate asks', async () => {
  // The counter-proof. Without it a predicate that says "no" to everything would
  // pass the test above while silencing a demand that was useful.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'with-material', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
      anchor('call-s'),
      { role: 'user', content: `more work ${'Q'.repeat(400)}` },
    ];
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const demands = gateOf(await round(hooks, ctx, messages));
    assert.equal(demands.length, 1, 'the episode holds material: the gate must ask to close it');
    assert.match(String(demands[0].content), /delimiter/);
  } finally { home.restore(); sandbox.cleanup(); }
});
