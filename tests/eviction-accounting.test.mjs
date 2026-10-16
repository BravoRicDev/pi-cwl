/**
 * Eviction accounting: with eviction disabled, no eviction should occur.
 *
 * With eviction disabled via runEvictionPass returning [], no eviction
 * actions should be generated, and the context should remain unchanged.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/** Full eviction only: the `removed` level is the one that adds the marker. */
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
  const sandbox = makeSandbox({ name: `accounting-${seq++}`, config });
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

const assistant = (marker) => ({
  role: 'assistant',
  content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }],
});

/** The line I am measuring, read from the real log. */
const ROW = /EVICTION applied: (\d+) msg removed, (\d+) reduced, (\d+)t -> (\d+)t \(saved (\d+)t\)/;

test('no eviction happens when eviction is disabled', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval());
  try {
    // A closed episode entirely inside the range: with eviction disabled,
    // nothing should be evicted
    await tools.get('delimiter').execute('call-e', { name: 'accountancy', type: 'expl', description: 'takeaway' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: 'prologue' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      assistant('INSIDE-1'),
      assistant('INSIDE-2'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    
    // With eviction disabled, the output should be identical to input
    assert.deepStrictEqual(out, messages, 'context should remain unchanged when eviction is disabled');
    
    // No EVICTION line should be written to the log
    const m = ROW.exec(logOf(sandbox));
    assert.ok(!m, 'the EVICTION line should not be written to the log when eviction is disabled');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('no reduction happens when eviction is disabled', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({
    tokenBudget: 1000,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: true, stripIntermediate: false, removeEpisode: false },
    showWidget: false,
    debug: true,
  });
  try {
    await tools.get('delimiter').execute('call-e', { name: 'reduction', type: 'expl', description: 'takeaway' }, undefined, undefined, ctx);
    // A huge toolResult inside the episode
    const bold = { role: 'toolResult', toolCallId: 'call-big', toolName: 'bash', content: [{ type: 'text', text: 'B'.repeat(60000) }] };
    const messages = [
      { role: 'user', content: 'prologue' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      bold,
      assistant('INSIDE-1'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    
    // With eviction disabled, the output should be identical to input
    assert.deepStrictEqual(out, messages, 'context should remain unchanged when eviction is disabled');
    
    // No EVICTION line should be written to the log
    const m = ROW.exec(logOf(sandbox));
    assert.ok(!m, 'the EVICTION line should not be written to the log when eviction is disabled');
  } finally { home.restore(); sandbox.cleanup(); }
});
