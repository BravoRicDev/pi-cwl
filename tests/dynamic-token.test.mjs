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
    // 1 prompt + 1 assistant with usage (prefix = 8000 tokens, but message chars are small)
    // This simulates real-world where system prompt + tools take most of the prompt.
    const messages = [
      user(1),
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

test('Option B: gate stays withheld when context exceeds budget only due to incompressible overhead', async () => {
  // tokenBudget is 5000, but system overhead is 6000 and messages are only ~200 tokens
  const { sandbox, home, hooks, ctx } = await boot({ tokenBudget: 5000 });
  try {
    const messages = [
      user(1),
      withUsage(200, { cacheRead: 6000, cacheWrite: 0, input: 500, output: 50 }),
    ];

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
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
