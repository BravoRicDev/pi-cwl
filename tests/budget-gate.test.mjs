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

test('with an episode recorded the gate asks, and proposes only the available option', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    // One recorded episode is enough for the gate to arm (CWL is in use).
    await tools.get('delimiter').execute('call-e', { name: 'work-in-progress', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
      { role: 'assistant', content: `reply ${'R'.repeat(400)}` },
      { role: 'user', content: `more ${'Q'.repeat(400)}` },
    ];
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const r = await round(hooks, ctx, messages);
    const demands = gateOf(r);
    assert.equal(demands.length, 1, 'the gate must ask to record a segment');
    const text = String(demands[0].content);
    assert.match(text, /delimiter/, 'it must propose recording a segment');
    assert.doesNotMatch(text, /cwl_compress_range/,
      'it must not propose an option that is not available (no compressible range)');
  } finally { home.restore(); sandbox.cleanup(); }
});

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

test('when the gate is armed but no segment can be recorded, it withholds and says why', async () => {
  // The gate arms as soon as CWL is in use, but it must still refuse to ask for the
  // impossible. With ONE message there is no segment a new closing could bound
  // (`canClose` needs a pair of messages) and no compressible range either. The demand
  // is WITHHELD, and the reason is written to the log: a silence without a reason is
  // the same defect one level down.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10, debug: true }));
  try {
    await tools.get('delimiter').execute('call-e', { name: 'tiny', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
    ];
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0,
        `turn ${i}: the gate asked for a segment that cannot exist`);
    }
    assert.match(logOf(sandbox), /GATE withheld: no segment can be recorded/,
      'the withheld demand was not explained in the log');
    // A demand that was never DELIVERED is not a failure of the agent.
    assert.doesNotMatch(logOf(sandbox), /GATE dropped/,
      'the gate gave up on a demand it never delivered');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the demand teaches the SINGLE-CALL contract: no action=start/end', async () => {
  // The counter-proof, and a guard on the contract: with implicit opens the demand must
  // NAME the one call that exists. A leftover `action="end"` would send the agent to a
  // call whose parameter no longer exists.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    await tools.get('delimiter').execute('call-e', { name: 'work', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: `question ${'P'.repeat(400)}` },
      { role: 'assistant', content: `reply ${'R'.repeat(400)}` },
      { role: 'user', content: `more work ${'Q'.repeat(400)}` },
    ];
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const demands = gateOf(await round(hooks, ctx, messages));
    assert.equal(demands.length, 1, 'the material is there: the gate must ask');
    const text = String(demands[0].content);
    assert.match(text, /delimiter\(name=/, 'the demand must name the single-call contract');
    assert.doesNotMatch(text, /action=/, 'the opening/closing action no longer exists');
  } finally { home.restore(); sandbox.cleanup(); }
});
