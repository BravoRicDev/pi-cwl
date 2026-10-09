/**
 * Level A: the budget follows the MODEL'S WINDOW.
 *
 * THE DEFECT. `tokenBudget` is one number, written for the biggest model the operator owns.
 * MEASURED on a 100k window: the fixed overhead (system prompt + tool schemas) is ~42.000t —
 * 42% of that window — and CWL cannot compress it, while Pi compacts at
 * `contextWindow - reserveTokens` (16.384 by default) on the WHOLE context: 83.616t, which is
 * ~41.565t of HISTORY. A 100.000t history budget on a 100k window is therefore UNREACHABLE:
 * Pi compacts first, summarises with a summary that knows nothing about the index, and the
 * graph this extension built is invalidated. A percentage alone would not have fixed it
 * either — a 60% share of the window is 60.000t of history and still above 41.565t — which is
 * why the fixed overhead is SUBTRACTED and the share is applied to the WHOLE context.
 *
 * THE CONTRACT PINNED HERE:
 *  1. `min(tokenBudget, share)` — the window can only LOWER the budget, never raise it. A
 *     configuration is a decision, and these tests drive the extension with `tokenBudget: 100`.
 *  2. No window (an older Pi, or no model resolved) → the CONFIGURED budget, unchanged: the
 *     feature degrades to the previous behaviour instead of to an invented number.
 *  3. The window is read LIVE, every turn: switching model must not need a restart.
 *
 * HOW THE TRIGGER IS OBSERVED. Not by reading state, and not by trusting that an eviction
 * happened: the `RANGE` row that `storeRange` writes on every turn carries the number the
 * decision compared against (`vs trigger Nt`), so the assertion is on the number that
 * actually decides.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx, withWindow, status } from './_helpers.mjs';

let seq = 0;

/** The configured budget is 100.000: the point is that a small window must BEAT it. */
const config = (extra = {}) => ({
  tokenBudget: 100_000,
  thresholdRatio: 1,
  dynamicBudgetRatio: 0.6,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(extra = {}) {
  const sandbox = makeSandbox({ name: `dyn-budget-${seq++}`, config: config(extra) });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => {
  try {
    return fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');
  } catch {
    return '';
  }
};

/**
 * The trigger of the LAST row that carried one — the number that decided.
 *
 * The rows are written in two shapes, and BOTH carry the trigger: `CONTEXT 180t under
 * threshold 76800t: no eviction` when the context is below it, and `... still above trigger
 * 76800t` when it is above. Matching only one of the two was the first version of this
 * helper, and it reported "no row carried a trigger" while the log was saying the right
 * number two lines above: a mute probe is a defect of the probe, not an absence.
 *
 * It fails LOUDLY when no row carried a trigger, rather than skipping: a suite that passes
 * because the observation never happened is the defect this file is written against.
 */
function lastTrigger(sandbox) {
  const found = [...logOf(sandbox).matchAll(/(?:threshold|trigger) (\d+)t/g)].map((m) => Number(m[1]));
  assert.ok(found.length > 0, `no row in the log carried a trigger. Log tail: ${logOf(sandbox).slice(-500)}`);
  return found[found.length - 1];
}

const conversation = () => [
  { role: 'user', content: `domanda ${'P'.repeat(200)}` },
  { role: 'assistant', content: `risposta ${'R'.repeat(200)}` },
  { role: 'user', content: `altra ${'Q'.repeat(200)}` },
];

async function turn(hooks, ctx, messages = conversation()) {
  await hooks.get('context')({ messages }, ctx);
  await hooks.get('turn_end')({}, ctx);
  return messages;
}

/**
 * The calibrated fixed overhead, read from the CACHE row.
 *
 * NOT from the state file: a fixture that never compressed has no `.pi/cwl/state` on disk at
 * all — MEASURED as `ENOENT: no such file or directory, scandir .../.pi/cwl/state` — so the
 * state was the wrong probe for this test, not a wrong value in it. The `CACHE` row prints
 * the calibrated overhead on every turn that sees a usage report.
 */
function lastOverhead(sandbox) {
  const found = [...logOf(sandbox).matchAll(/overhead=(\d+)t/g)].map((m) => Number(m[1]));
  assert.ok(found.length > 0, `no CACHE row carried an overhead. Log tail: ${logOf(sandbox).slice(-500)}`);
  return found[found.length - 1];
}

const user = (i) => ({ role: 'user', content: `prompt ${i} ${'P'.repeat(200)}` });
const withUsage = (ts, usage) => ({ role: 'assistant', content: 'answer body', timestamp: ts, usage });

test('finestra enorme: il 60% non morde e il budget resta quello configurato', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 1_000_000);
    await turn(hooks, ctx);
    assert.equal(
      lastTrigger(sandbox),
      100_000,
      'su 1M il 60% vale 600.000: il min deve scegliere i 100.000 configurati',
    );
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('finestra 200k: il 60% (120.000) sta ancora sopra il configurato, quindi vince il configurato', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 200_000);
    await turn(hooks, ctx);
    assert.equal(lastTrigger(sandbox), 100_000, '60% di 200k = 120.000 > 100.000: il min non deve alzare');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('finestra 128k: il cap morde e il trigger scende a 76.800', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 128_000);
    await turn(hooks, ctx);
    assert.equal(lastTrigger(sandbox), 76_800, '60% di 128.000 = 76.800');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('finestra 100k: il trigger scende a 60.000 — sotto gli ~83.616t totali in cui Pi compatterebbe', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 100_000);
    await turn(hooks, ctx);
    assert.equal(lastTrigger(sandbox), 60_000, '60% di 100.000 = 60.000 (overhead non ancora calibrato)');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('il min non ALZA mai il budget sopra quello configurato', async () => {
  const { sandbox, home, hooks, ctx } = await boot({ tokenBudget: 5_000 });
  try {
    withWindow(ctx, 1_000_000);
    await turn(hooks, ctx);
    assert.equal(
      lastTrigger(sandbox),
      5_000,
      'una finestra enorme NON deve alzare un budget configurato piccolo: e\' un min, non un max',
    );
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('senza finestra il budget e\' quello configurato: il caso Pi vecchio non cambia niente', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    await turn(hooks, ctx);
    assert.equal(
      lastTrigger(sandbox),
      100_000,
      'nessun getContextUsage e nessun modello: comportamento identico a prima della modifica',
    );
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('se getContextUsage manca, la finestra arriva dal modello: il ripiego e\' usato davvero', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 128_000, { source: 'model' });
    assert.equal(typeof ctx.getContextUsage, 'undefined', 'il fixture deve esporre SOLO ctx.model');
    await turn(hooks, ctx);
    assert.equal(lastTrigger(sandbox), 76_800, 'la seconda fonte deve dare la stessa risposta della prima');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('la finestra cambia fra due turni e il trigger cambia con lei: nessun valore congelato', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    withWindow(ctx, 1_000_000);
    await turn(hooks, ctx);
    assert.equal(lastTrigger(sandbox), 100_000, 'primo turno: finestra enorme, cap inerte');

    withWindow(ctx, 100_000);
    await turn(hooks, ctx);
    assert.equal(
      lastTrigger(sandbox),
      60_000,
      'secondo turno: la finestra e\' cambiata e il trigger deve seguirla senza riavvio',
    );
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test("l'overhead fisso si SOTTRAE dalla quota della finestra", async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    // Calibration: the provider prefix is 8000t while the message chars are small, so the
    // difference IS the incompressible overhead. Same fixture as dynamic-token.test.mjs.
    const outgoing = [user(1)];
    await hooks.get('context')({ messages: outgoing }, ctx);
    const messages = [...outgoing, withUsage(100, { cacheRead: 7500, cacheWrite: 0, input: 500, output: 100 })];
    withWindow(ctx, 128_000);
    await hooks.get('context')({ messages }, ctx);

    // The overhead comes from the LOG, not from the state file: see `lastOverhead`.
    const overhead = lastOverhead(sandbox);
    assert.ok(
      overhead > 0,
      `the fixture did not calibrate the overhead, so this test would prove nothing about the subtraction. Log tail: ${logOf(sandbox).slice(-400)}`,
    );

    // THE LAG IS PART OF THE CONTRACT, and the first version of this test got it wrong: the
    // calibrated overhead of turn 1 is written by the CACHE row, which runs AFTER the budget
    // decision — the same one-turn lag the extension documents for `providerContextTokens`
    // ("the freshest measurement at the top of the hook is the previous turn's"). So turn 1
    // measured 7940t while still deciding with 76.800t, and asserting the subtraction on that
    // same turn was asserting something the design does not promise. Turn 2 is where it lands.
    await hooks.get('context')({ messages }, ctx);
    assert.equal(
      lastTrigger(sandbox),
      76_800 - overhead,
      `on the turn AFTER calibration the share must be REDUCED by the fixed overhead (measured ${overhead}t)`,
    );
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('cwl_status pubblica il budget effettivo, la finestra e la fonte, col configurato intatto', async () => {
  const { sandbox, home, hooks, tools, ctx } = await boot();
  try {
    withWindow(ctx, 128_000);
    await turn(hooks, ctx);
    const st = await status(tools, ctx);

    assert.equal(st.details.effectiveBudget, 76_800, 'il budget in forza deve essere quello effettivo');
    assert.equal(st.details.contextWindow, 128_000, 'la finestra osservata deve essere pubblicata');
    assert.equal(st.details.budgetSource, 'window', 'la fonte deve dire da dove viene il numero');
    assert.equal(
      st.details.budget,
      100_000,
      'il campo `budget` resta il CONFIGURATO: e\' un contratto pubblicato, non una rinomina',
    );
    assert.match(st.text, /128[.,]000/, `lo status deve mostrare la finestra. Testo: ${st.text}`);
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('cwl_status dichiara la fonte "configured" quando nessuna finestra e\' nota', async () => {
  const { sandbox, home, hooks, tools, ctx } = await boot();
  try {
    await turn(hooks, ctx);
    const st = await status(tools, ctx);
    assert.equal(st.details.budgetSource, 'configured');
    assert.equal(st.details.effectiveBudget, 100_000);
    assert.equal(st.details.contextWindow, 0, '0 significa "mai visto", non una finestra inventata');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
