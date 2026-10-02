import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;
const config = {
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  looseLeaves: 0,
  nodeCapacity: 2,
  mergeNodesAt: 2,
  mergeMinRatio: 0,
  mergeMinChars: 0,
};
const tick = () => new Promise((resolve) => setTimeout(resolve, 10));
const conversation = (from = 1, to = 12) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `user-${i} ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `assistant-${i} ` + 'A'.repeat(200) });
  }
  return out;
};
const stateFile = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((name) => name.endsWith('.json'));
  assert.ok(file, 'session state should be persisted');
  return path.join(dir, file);
};
const readState = (sandbox) => JSON.parse(fs.readFileSync(stateFile(sandbox), 'utf8'));
const okSummary = (summary) => ({ stopReason: 'stop', content: [{ type: 'text', text: JSON.stringify({ summary }) }] });

async function setup(complete) {
  const sandbox = makeSandbox({ name: `pit-background-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  const model = { provider: 'test-provider', id: 'pit-summarizer' };
  let authEnabled = false;
  ctx.model = model;
  ctx.modelRegistry = {
    find: (provider, id) => provider === model.provider && id === model.id ? model : null,
    hasConfiguredAuth: () => authEnabled,
    complete,
  };
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx, enableAuth: () => { authEnabled = true; } };
}

async function prepareDuePit({ tools, hooks, ctx }) {
  for (let i = 1; i <= 4; i++) {
    // With looseLeaves: 0 and nodeCapacity: 2, 4 leaves will form 2 nodes.
    await hooks.get('context')({ messages: conversation(1, i + 3) }, ctx);
    const result = await tools.get('cwl_compress_range').execute(
      'test', { summary: `BODY-${i} ` + 'x'.repeat(300), micro: `MICRO-${i} ` + 'm'.repeat(80) }, undefined, undefined, ctx,
    );
    assert.equal(result.details.ok, true, `leaf ${i}: ${JSON.stringify(result.details)}`);
  }
  // Now we have 4 leaves = 2 nodes. mergeNodesAt: 2.
}

test('a due pit is summarized asynchronously and applied atomically without a cwl_old call', async () => {
  let calls = 0;
  let resolve;
  const { sandbox, home, tools, hooks, ctx, enableAuth } = await setup(() => {
    calls++;
    return new Promise((done) => { resolve = done; });
  });
  try {
    await prepareDuePit({ tools, hooks, ctx });
    enableAuth();
    // Trigger the job.
    await hooks.get('context')({ messages: conversation() }, ctx);
    await tick();
    assert.equal(calls, 1, 'the context hook should launch one background completion');
    const pending = readState(sandbox).backgroundPitSummary;
    assert.equal(pending.status, 'pending');
    assert.equal(pending.attempts, 1);
    resolve(okSummary('Automatic pit synthesis: stories one and two.'));
    await tick();
    await tick();
    await hooks.get('context')({ messages: conversation() }, ctx);
    await tick();
    await tick();
    await tick();
    const state = readState(sandbox);
    assert.equal(state.oldNode.summary, 'Automatic pit synthesis: stories one and two.');
    assert.notEqual(state.backgroundPitSummary?.fingerprint, pending.fingerprint,
      'after applying one batch, the next due batch may be scheduled independently');
    assert.ok(state.oldNode, 'automatic completion should create/update the pit');
    assert.ok(state.oldNode.nodes.length > 0);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a ready pit summary is discarded if the pit inputs changed while the model was working', async () => {
  let resolve;
  let calls = 0;
  const { sandbox, home, tools, hooks, ctx, enableAuth } = await setup(() => {
    calls++;
    return new Promise((done) => { resolve = done; });
  });
  try {
    await prepareDuePit({ tools, hooks, ctx });
    enableAuth();
    await hooks.get('context')({ messages: conversation() }, ctx);
    await tick();
    assert.equal(calls, 1);
    // Change a micro represented by the requested node while the completion is pending.
    const state = readState(sandbox);
    const sourceNode = state.nodes.find((node) => node.id === state.backgroundPitSummary.nodes[0]);
    assert.ok(sourceNode);
    const leafId = sourceNode.leaves[0];
    const changed = await tools.get('cwl_micro').execute('test', { id: leafId, text: 'changed while summarizing' }, undefined, undefined, ctx);
    assert.equal(changed.details.ok, true);
    resolve(okSummary('STALE synthesis'));
    await tick();
    await tick();
    await tick();
    await hooks.get('context')({ messages: conversation() }, ctx);
    const after = readState(sandbox);
    assert.equal(after.oldNode, null, 'stale result must not create a pit');
    assert.ok(after.backgroundPitSummary, 'a changed pit may be rescheduled against its new fingerprint');
    assert.notEqual(after.backgroundPitSummary.fingerprint, state.backgroundPitSummary.fingerprint,
      'the changed source must not reuse the stale request fingerprint');
    assert.doesNotMatch(JSON.stringify(after.oldNode), /STALE synthesis/);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('two provider failures mark the background request exhausted and leave the manual tool available', async () => {
  let calls = 0;
  const { sandbox, home, tools, hooks, ctx, enableAuth } = await setup(async () => {
    calls++;
    throw new Error('provider unavailable');
  });
  try {
    await prepareDuePit({ tools, hooks, ctx });
    enableAuth();
    await hooks.get('context')({ messages: conversation() }, ctx);
    await tick();
    await tick();
    await tick();
    await tick();
    const state = readState(sandbox);
    assert.equal(calls, 2);
    assert.equal(state.backgroundPitSummary.status, 'exhausted');
    assert.equal(state.backgroundPitSummary.attempts, 2);
    const manual = await tools.get('cwl_old').execute(
      'test', { text: 'Manual fallback synthesis.' }, undefined, undefined, ctx,
    );
    assert.equal(manual.details.ok, true, JSON.stringify(manual.details));
    assert.equal(readState(sandbox).oldNode.summary, 'Manual fallback synthesis.');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});
