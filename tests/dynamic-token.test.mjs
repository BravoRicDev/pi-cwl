import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx, status } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 10_000,
  thresholdRatio: 1,
  protectedTurns: 2,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(extra = {}) {
  const sandbox = makeSandbox({ name: `dyn-token-${seq++}`, config: config(extra) });
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

const user = (i) => ({ role: 'user', content: `prompt ${i} ${'P'.repeat(200)}` });
const withUsage = (ts, usage) => ({
  role: 'assistant',
  content: 'answer body',
  timestamp: ts,
  usage,
});

test('online calibration computes charTokenRatio and systemOverheadTokens from assistant usage', async () => {
  const { sandbox, home, hooks, tools, ctx } = await boot();
  try {
    // Measure the exact outgoing prompt first; the following assistant usage belongs to it.
    // The provider prefix is 8000 tokens while the message chars are small, simulating
    // the system prompt + tools that are absent from Pi's message list.
    const outgoing = [user(1)];
    await hooks.get('context')({ messages: outgoing }, ctx);
    const messages = [
      ...outgoing,
      withUsage(100, { cacheRead: 7500, cacheWrite: 0, input: 500, output: 100 }),
    ];

    await hooks.get('context')({ messages }, ctx);
    const log = logOf(sandbox);

    // CACHE line must log ratio and overhead
    assert.match(log, /ratio=\d+\.\d{2} overhead=\d+t/, `cache line did not log ratio/overhead: ${log}`);

    // cwl_status must report measured context tokens with overhead and ratio
    const st = await status(tools, ctx);
    assert.ok(st.text.includes('overhead fisso:') || st.text.includes('fixed overhead:'),
      `cwl_status did not mention overhead: ${st.text}`);
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('assistant usage is paired with the exact preceding outgoing prompt snapshot', async () => {
  const { sandbox, home, hooks, tools, ctx } = await boot();
  try {
    const firstPrompt = [user(1)];
    await hooks.get('context')({ messages: firstPrompt }, ctx);

    const response = withUsage(100, { cacheRead: 7500, cacheWrite: 0, input: 500, output: 100 });
    const nextPrompt = [...firstPrompt, response, { role: 'user', content: `next ${'N'.repeat(1600)}` }];
    await hooks.get('context')({ messages: nextPrompt }, ctx);

    let log = logOf(sandbox);
    const expectedFirstChars = JSON.stringify(firstPrompt[0]).length;
    const firstRow = log.split('\n').find((line) => line.includes('CACHE ') && line.includes('sample=paired'));
    assert.ok(firstRow, `no paired usage sample was logged: ${log}`);
    assert.ok(firstRow.includes(`promptChars=${expectedFirstChars} providerPrefix=8000`),
      `usage was not paired with the first outgoing prompt (${expectedFirstChars} chars): ${firstRow}`);
    assert.ok(!firstRow.includes(`promptChars=${JSON.stringify(nextPrompt).length}`),
      'calibration used the current response-bearing list instead of the measured outgoing request');

    // A second paired sample with a larger prompt calibrates only the delta between
    // the two exact request snapshots, never the response list that carries usage.
    const secondResponse = withUsage(200, { cacheRead: 8100, cacheWrite: 0, input: 500, output: 120 });
    const thirdPrompt = [...nextPrompt, secondResponse];
    await hooks.get('context')({ messages: thirdPrompt }, ctx);
    log = logOf(sandbox);
    const cacheRows = log.split('\n').filter((line) => line.includes('CACHE ') && line.includes('sample=paired'));
    assert.equal(cacheRows.length, 2, `expected two distinct paired usage rows: ${cacheRows.join('\n')}`);
    const expectedSecondChars = nextPrompt.reduce((sum, message) => sum + JSON.stringify(message).length, 0);
    assert.ok(cacheRows[1].includes(`promptChars=${expectedSecondChars} providerPrefix=8600`),
      `the second usage sample was not paired with the second outgoing request: ${cacheRows[1]}`);
    const ratio = Number(/ratio=(\d+\.\d+)/.exec(cacheRows[1])?.[1]);
    assert.ok(ratio > 1.8 && ratio < 4, `the comparable delta did not calibrate the default ratio: ${cacheRows[1]}`);

    const st = await status(tools, ctx);
    assert.ok(st.text.includes('overhead fisso:') || st.text.includes('fixed overhead:'),
      `cwl_status did not mention overhead: ${st.text}`);
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('an empty context callback prevents stale request snapshots from being paired', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    await hooks.get('context')({ messages: [user(1)] }, ctx);
    await hooks.get('context')({ messages: [] }, ctx);
    const messages = [user(1), withUsage(100, { cacheRead: 7500, cacheWrite: 0, input: 500, output: 100 })];
    await hooks.get('context')({ messages }, ctx);
    const log = logOf(sandbox);
    assert.match(log, /CACHE .*sample=unpaired promptChars=0 providerPrefix=8000/,
      `the usage report was incorrectly paired with an obsolete snapshot: ${log}`);
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('Option B: gate stays withheld when context exceeds budget only due to incompressible overhead', async () => {
  // tokenBudget is 5000, but system overhead is 6000 and messages are only ~200 tokens
  const { sandbox, home, hooks, ctx } = await boot({ tokenBudget: 5000 });
  try {
    const messages = [
      user(1),
      withUsage(200, { cacheRead: 6000, cacheWrite: 0, input: 500, output: 50 }),
    ];

    // This exact outgoing prompt is the one measured by the assistant usage below.
    await hooks.get('context')({ messages: [user(1)] }, ctx);

    // Arms the gate over budget
    await hooks.get('turn_end')({}, ctx);
    await hooks.get('turn_end')({}, ctx);
    await hooks.get('turn_end')({}, ctx);

    const out = await hooks.get('context')({ messages }, ctx);
    // Gate must NOT inject a demand because there is no compressible arbitrio
    if (out && out.messages) {
      const injectedGate = out.messages.some((m) => m.customType === 'cwl-budget-gate');
      assert.equal(injectedGate, false, 'gate demanded compression even though messages were within limit and arbitrio is 0');
    }

    const log = logOf(sandbox);
    assert.match(log, /GATE withheld:.*due to incompressible system overhead|no compressible range/,
      `expected GATE withheld log, got: ${log}`);
    assert.match(log, /CACHE .*sample=paired .*overhead=\d+t/, `provider overhead was not measured from the matching prompt: ${log}`);
    assert.match(log, /history \(~\d+t\) is within limits/, `history should remain below the trigger independently of fixed overhead: ${log}`);
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
