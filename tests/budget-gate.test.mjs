/**
 * Level B: the gate that USED TO ask the AGENT to compact.
 *
 * WHAT IT WAS. Deterministic eviction is blind: it cuts by age, not by meaning. The gate
 * was the only channel that could get a semantic decision out of the model, and it did it
 * by appending a custom message to the end of every turn, naming a call to make. The gate
 * closed by EFFECT and not by echo — an invented confirmation is worth nothing, only the
 * measured context counts — and it was never a guarantee: the Level A parachute is what
 * actually holds the context.
 *
 * WHAT IT IS NOW, and this is the decision these tests pin. MEASURED in a live session:
 * that message rewrote the tail of the prompt on EVERY turn, it cost tokens in the currency
 * the session was short of, and it made the agent do work the extension can do by itself.
 * The operator removed it: "togliere qualsiasi avviso che chiede all'agente di comprimere".
 * The substitute is the background job scheduled a few lines above the old injection — the
 * summarizer that writes the micro together with the summary — plus the deterministic
 * eviction. And since no demand is ever delivered, `turn_end` can never charge the agent for
 * an attempt it was not given: the counter stays at zero and the gate has nothing to give up on.
 *
 * So the gate still EXISTS and still ARMS: it is the thing that decides WHEN the automatic
 * compaction fires. What it must never do again is speak.
 *
 * Three promises are pinned below:
 *  1. no `cwl-budget-gate` message ever reaches the context, in the very scenarios that
 *     used to produce one;
 *  2. the gate still ARMS (`st.gateArmedTurn`), and when it cannot do anything it says WHY
 *     in the log — a silence without a reason is the same defect one level down;
 *  3. the two new mechanisms that replaced it behave: the HYSTERESIS latch and the
 *     auto-widen of the protected window.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// No active level: so NOTHING gets compacted deterministically and the context stays above
// budget, which is the condition in which the gate must act.
const overBudgetConfig = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  gate: true,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
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

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const stateOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

/** A full round: context hook + end of turn. */
async function round(hooks, ctx, messages) {
  const res = await hooks.get('context')({ messages }, ctx);
  await hooks.get('turn_end')({}, ctx);
  return res;
}

/**
 * The gate arms only when it has something the agent COULD do (`actionable` =
 * `rangeStartHash !== null || graph.count > 0`). A segment recorded with `delimiter` is the
 * cheapest way to be in that state, and `protectedTurns: 10` keeps the material protected so
 * the fixture does not collapse under its own eviction.
 */
/** The tokens the LAST floor row measured for a total, and the trigger it was compared to. */
function measuredTotal(fragment) {
  const rows = [...fragment.matchAll(/CONTEXT (\d+)t estimated total still above trigger (\d+)t/g)];
  if (rows.length === 0) return null;
  const last = rows[rows.length - 1];
  return { tokens: Number(last[1]), trigger: Number(last[2]) };
}

/**
 * The size of the payload that must land INSIDE the hysteresis band. The band is
 * (trigger, trigger * 1.15] = (50, 57.5] tokens here, and the estimate is not `chars / 4`:
 * MEASURED, `estimateMessageListTokens` adds roughly 7.75t of per-message overhead, so the
 * nominal 54t of 216 characters is really about 62t — ABOVE the band, which would test a
 * second closing instead of the hold. The test verifies where it landed and fails loudly
 * rather than adjusting itself to whatever the code happens to do.
 */
const BAND_CHARS = 180;

async function armedFixture(extra = {}) {
  const booted = await boot(overBudgetConfig({ protectedTurns: 10, ...extra }));
  await booted.tools.get('delimiter').execute(
    'call-e', { name: 'work-in-progress', type: 'act' }, undefined, undefined, booted.ctx,
  );
  return booted;
}

const shortConversation = () => [
  { role: 'user', content: `question ${'P'.repeat(400)}` },
  { role: 'assistant', content: `reply ${'R'.repeat(400)}` },
  { role: 'user', content: `more ${'Q'.repeat(400)}` },
];

test('the gate NEVER writes into the context, and it still ARMS and says why it is silent', async () => {
  const { sandbox, home, hooks, ctx } = await armedFixture();
  try {
    const messages = shortConversation();
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(
        gateOf(r).length,
        0,
        `turn ${i + 1}: the gate is no longer a channel — nothing may be injected. Messages: ${
          (r?.messages ?? []).map((m) => m.customType || m.role).join(', ')}`,
      );
    }

    // The gate is not dead: it armed. Losing this would mean the automatic compaction
    // never fires either, which is the real failure hiding behind a silent channel.
    const st = stateOf(sandbox);
    assert.ok(
      st.gateArmedTurn >= 0,
      `the gate never armed: the channel went silent AND the automatic compaction was never triggered. State: ${JSON.stringify(
        { gateArmedTurn: st.gateArmedTurn, overBudgetSince: st.overBudgetSince, evicting: st.evicting },
      )}`,
    );

    // Nobody is charged for an attempt they were never given, and the COUNTER is now the only
    // thing there is to assert: the flag that used to record "this turn's demand was really
    // delivered" is GONE, together with the one assignment that could ever set it, because it
    // could only ever be false. Asserting on a removed field would be asserting on
    // `undefined`, which is not the same as asserting that no attempt was made.
    assert.equal(st.gateAttempts, 0, 'the attempt counter moved without a demand being shown');

    const log = logOf(sandbox);
    assert.match(log, /GATE armed at turn \d+/, 'the arming was not recorded in the log');

    // And the silence is EXPLAINED. Either the gate withheld with a reason, or it went
    // through to the by-design row. Both are acceptable; a bare silence is not.
    assert.ok(
      /GATE withheld: .+/.test(log) || /the channel is silent by design/.test(log),
      `the gate stayed silent without saying why: a silence without a reason is the same defect one level down. Log tail: ${
        log.trim().split('\n').slice(-5).join(' | ')}`,
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('gate=false: the channel is off and the gate never even arms', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ gate: false, protectedTurns: 10 }));
  try {
    // The delimiter is not decoration: `saveState` only writes when the session has USED CWL
    // (a recorded segment or a leaf), so without it there is no state file to read and the
    // assertion below would be testing the file system instead of the gate.
    await tools.get('delimiter').execute('call-g', { name: 'off', type: 'act' }, undefined, undefined, ctx);
    const messages = heavy();
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0, 'gate=false must disable the channel entirely');
    }
    assert.equal(stateOf(sandbox).gateArmedTurn, -1, 'gate=false must not arm the gate');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('under budget the gate stays off and the latch stays open', async () => {
  // 100_000 with a ratio of 0.5 is a trigger of 50_000: 6 heavy messages are nowhere near it.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig({ tokenBudget: 100_000 }));
  try {
    // See the note in the test above: the state file only exists once CWL has been used.
    await tools.get('delimiter').execute('call-u', { name: 'under', type: 'act' }, undefined, undefined, ctx);
    const messages = heavy();
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0, 'under budget nothing may be injected');
    }
    const st = stateOf(sandbox);
    assert.equal(st.gateArmedTurn, -1, 'under budget the gate must not arm');
    assert.equal(st.evicting, false, 'under budget the latch must be open');
    assert.equal(st.overBudgetSince, -1, 'under budget the "over since" marker must be cleared');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the latch is HYSTERESIS: inside the band it HOLDS instead of crossing twice', async () => {
  // Trigger = 100 * 0.5 = 50 tokens; the latch closes above 50 * 1.15 = 57.5 and reopens
  // below 50. Between the two the previous decision stands — that is the whole point of the
  // latch, and the reason a turn of stale measurement cannot make it oscillate.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig());
  try {
    // See the note in the tests above: the state file only exists once CWL has been used.
    await tools.get('delimiter').execute('call-l', { name: 'latch', type: 'act' }, undefined, undefined, ctx);
    // Way above: it closes.
    await round(hooks, ctx, heavy());
    assert.equal(stateOf(sandbox).evicting, true, 'far above the trigger the latch must close');

    // Inside the band, and it was CLOSED: it must stay closed. Where it landed is MEASURED
    // back from the log, not assumed from the character count.
    const bandBefore = logOf(sandbox).length;
    await round(hooks, ctx, [{ role: 'user', content: 'P'.repeat(BAND_CHARS) }]);
    const inBand = measuredTotal(logOf(sandbox).slice(bandBefore));
    assert.ok(inBand, 'the payload is not above the trigger at all, so it cannot test the band — raise BAND_CHARS');
    assert.ok(
      inBand.tokens > inBand.trigger && inBand.tokens <= inBand.trigger * 1.15,
      `the fixture is NOT inside the band (${inBand.tokens}t against a trigger of ${inBand.trigger}t, band ends at `
        + `${inBand.trigger * 1.15}t): it would be testing a second closing instead of the hold. Adjust BAND_CHARS.`,
    );
    assert.equal(
      stateOf(sandbox).evicting,
      true,
      'inside the band the latch must HOLD its previous value: closing and reopening inside the band is exactly the '
        + 'oscillation the latch exists to prevent',
    );

    // Well below: it reopens.
    await round(hooks, ctx, [{ role: 'user', content: 'short' }]);
    assert.equal(stateOf(sandbox).evicting, false, 'below the trigger the latch must reopen');

    // Inside the band again, and it was OPEN: it must stay open. Here the row's ABSENCE is the
    // evidence, and it is the only evidence that can exist: with the latch held OPEN the hook
    // takes the "not over budget" path and never reaches `declareFloor`, which is called only
    // once the latch has closed. That the payload really IS in the band was MEASURED on the
    // first pass with the same list — the estimate depends on the list and the ratio alone.
    const bandBefore2 = logOf(sandbox).length;
    await round(hooks, ctx, [{ role: 'user', content: 'P'.repeat(BAND_CHARS) }]);
    const rows2 = logOf(sandbox).slice(bandBefore2);
    assert.equal(
      measuredTotal(rows2),
      null,
      `the payload was treated as OVER the latch although it is inside the band and the latch was open: ${
        rows2.trim().split('\n').slice(-3).join(' | ')}`,
    );
    assert.equal(
      stateOf(sandbox).evicting,
      false,
      'inside the band the latch must HOLD the OPEN value too: it is a latch, not a second threshold',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the wide window is granted for a SINGLE turn and costs a turn every four', async () => {
  // The auto-widen is the substitute the operator chose: when the gate fires, the protected
  // window drops to one turn for THAT turn only, so the eviction has something to bite on.
  // It cannot fire on the first over-budget turn (`overBudgetSince` is still -1 there), and
  // it cannot fire twice inside AUTO_WIDEN_EVERY_TURNS turns.
  const { sandbox, home, hooks, ctx } = await armedFixture();
  try {
    // `heavy()`, and not a three-message conversation: `protectedFromIndex` has a DEGENERATE
    // case (fewer user turns in the list than the window asked for) that returns a share of
    // the list instead of the configured window, and with only two user turns that share is
    // the SAME floor for any window — so the widen would be invisible by construction. Six
    // user turns make the two windows genuinely different (~3 messages protected against 1).
    const messages = heavy();
    const seen = [];
    for (let i = 0; i < 8; i++) {
      await round(hooks, ctx, messages);
      const st = stateOf(sandbox);
      seen.push({ turn: i + 1, widen: st.autoWidenTurn, overSince: st.overBudgetSince });
    }

    const widened = seen.filter((s) => s.widen >= 0);
    assert.ok(
      widened.length >= 1,
      `the widen never fired in 8 over-budget turns, so the protected window is never opened and the eviction has ` +
        `nothing to act on: ${JSON.stringify(seen)}`,
    );

    // NEVER twice inside the cooldown window. `autoWidenTurn` PERSISTS its value into the
    // turns that FOLLOW the one it fired on, so the fires are the DISTINCT values of that
    // field: comparing consecutive rows would measure how long the flag stayed set.
    const fires = [...new Set(widened.map((s) => s.widen))];
    for (let i = 1; i < fires.length; i++) {
      assert.ok(
        fires[i] - fires[i - 1] >= 4,
        `the widen fired again after ${fires[i] - fires[i - 1]} turn(s): it may only cost a turn every 4. ${JSON.stringify(seen)}`,
      );
    }

    // The protected window really shrinks on the widen turn — the flag must not be inert.
    const rows = [...logOf(sandbox).matchAll(/still above trigger \d+t: (\d+)t in the protected window/g)]
      .map((m) => Number(m[1]));
    assert.ok(rows.length >= 3, `the protected-window rows are missing from the log: ${logOf(sandbox).slice(-400)}`);
    assert.ok(
      Math.min(...rows) < Math.max(...rows),
      `the widen changed the flag but not the window: every row measured the same protected quota (${JSON.stringify(rows)})`,
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('with NOTHING to do the gate does not ask: it must never ask the impossible', async () => {
  // Reproduces a real session. Measured there: the gate asked to compact, but
  //  - no episode was open (all 4 closed)  -> option 1 unavailable
  //  - cwl_compress_range answered "nothing left to compress"
  //    -> option 2 unavailable
  // So it asked for something impossible every turn, burning context. The demand is gone,
  // but the promise that survives is the one that matters: the gate must still refuse to
  // trigger work that cannot be done, and must say why it withheld.
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig({ protectedTurns: 10 }));
  try {
    const messages = [
      { role: 'user', content: 'question ' + 'P'.repeat(400) },
      { role: 'assistant', content: 'reply ' + 'R'.repeat(400) },
    ];
    for (let i = 0; i < 8; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(
        gateOf(r).length,
        0,
        `turn ${i}: the gate asked to compact without having anything to propose`,
      );
    }
    // WHAT KEEPS THE IMPOSSIBLE ASK AWAY HERE is not the withholding branch but the ARMING
    // guard: with no recorded segment and no stored range, `actionable` is false, so the gate
    // never arms at all — MEASURED on this very fixture: the log says
    // `RANGE source=hook-input none` and `GATE armed` never appears. Asserting the
    // `GATE withheld` row HERE would be asserting a branch the code never reaches; the real
    // guard is the one below, and the withholding branch has its own test further down, with
    // a segment recorded.
    //
    // The probe is the LOG and not the state file, and that is not a shortcut: `saveState`
    // only writes once the session has USED CWL, and this fixture deliberately has nothing
    // recorded — so there is no state file here, and an assertion on one would fail on the
    // file system instead of on the gate.
    assert.doesNotMatch(
      logOf(sandbox),
      /GATE armed/,
      'the gate armed while it had nothing to propose: it would then ask for something no call can do',
    );    // A demand that was never DELIVERED is not a failure of the agent.
    assert.doesNotMatch(logOf(sandbox), /GATE dropped/, 'the gate gave up on a demand it never delivered');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the gate withholds when no background job can start and nothing is compressible', async () => {
  // The case the test above does NOT cover, and the one that actually happened. There, a
  // SHORT list made the old predicate agree by accident. Here the list is LONG and the gate
  // is armed, but only one message is ELIGIBLE: an empty body hashes to sha256("") and is not
  // a valid endpoint, so `first === last` and `compressibleRange` returns null.
  //
  // The old guard was `canClose = list.length >= 2`, true for ANY real conversation, so the
  // demand fired on every turn asking for a `delimiter` that frees nothing. The honest
  // predicate is the scan `/cwl_save` already used, and with the channel silent what is left
  // to verify is that the gate still REFUSES to burn a turn on it.
  const { sandbox, home, tools, hooks, ctx } = await boot(overBudgetConfig());
  try {
    await tools.get('delimiter').execute('call-w', { name: 'work', type: 'act' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: '' },
      { role: 'user', content: `only ${'P'.repeat(400)}` },
    ];
    assert.equal(messages.length >= 2, true, 'the OLD predicate would have called this closable');
    for (let i = 0; i < 4; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(
        gateOf(r).length,
        0,
        `turn ${i + 1}: nothing is eligible, so no call can free anything — the gate must stay silent`,
      );
    }
    // Here the segment IS recorded, so the gate arms and the withholding branch IS reached:
    // the silence must come with a reason attached, or the next session pays the same
    // diagnosis again from scratch.
    assert.match(
      logOf(sandbox),
      /GATE withheld: .+/,
      `the gate went silent without saying why: ${logOf(sandbox).trim().split('\n').slice(-5).join(' | ')}`,
    );
  } finally { home.restore(); sandbox.cleanup(); }
});
