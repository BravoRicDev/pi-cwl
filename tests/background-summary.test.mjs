import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const config = {
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 2,
  gate: true,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
};

const conversation = () => {
  const messages = [];
  for (let turn = 1; turn <= 6; turn++) {
    messages.push({ role: 'user', content: `question ${turn} ` + 'U'.repeat(220), timestamp: turn * 2 });
    messages.push({ role: 'assistant', content: `answer ${turn} ` + 'A'.repeat(220), timestamp: turn * 2 + 1 });
  }
  return messages;
};

const tick = () => new Promise((resolve) => setImmediate(resolve));
const hasGate = (out) => (out?.messages ?? []).some((message) => message.customType === 'cwl-budget-gate');

async function setup(name, complete) {
  const sandbox = makeSandbox({ name, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  const model = { provider: 'test-provider', id: 'test-summarizer' };
  ctx.model = model;
  ctx.modelRegistry = {
    find: (provider, id) => provider === model.provider && id === model.id ? model : null,
    hasConfiguredAuth: () => true,
    complete,
  };
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

async function reachBackground(hooks, ctx, messages, getCalls) {
  for (let turn = 0; turn < 12; turn++) {
    const out = await hooks.get('context')({ messages }, ctx);
    await hooks.get('turn_end')({}, ctx);
    if (getCalls() > 0) return out;
  }
  assert.fail('the existing budget trigger did not start a background summary');
}

function persistedState(sandbox) {
  const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(stateDir).find((name) => name.endsWith('.json'));
  assert.ok(file, 'background work must persist a session state file');
  return JSON.parse(fs.readFileSync(path.join(stateDir, file), 'utf8'));
}

const result = (micro, summary) => ({
  stopReason: 'stop',
  content: [{ type: 'text', text: JSON.stringify({ micro, summary }) }],
});

test('the current gate starts a non-blocking background completion and CWL supplies leaf id and turn bounds', async () => {
  let calls = 0;
  let resolveCompletion;
  const { sandbox, home, hooks, ctx } = await setup('background-ready', (...args) => {
    calls++;
    const prompt = args[1].messages[0].content[0].text;
    assert.match(prompt, /Return ONLY one valid JSON object/);
    assert.match(prompt, /original turns 1-4/);
    return new Promise((resolve) => { resolveCompletion = resolve; });
  });
  const messages = conversation();
  try {
    const first = await reachBackground(hooks, ctx, messages, () => calls);
    assert.equal(calls, 1, 'the hook starts exactly one completion without waiting for it');
    assert.equal(hasGate(first), false, 'background work replaces the hook invitation');
    const pending = persistedState(sandbox).backgroundSummary;
    assert.equal(pending.status, 'pending');
    assert.equal(pending.attempts, 1, 'attempt is persisted before completion settles');

    resolveCompletion(result('Brief source label', 'Full source summary for resumption'));
    await tick();
    const applied = await hooks.get('context')({ messages }, ctx);
    const after = persistedState(sandbox);
    assert.ok(JSON.stringify(applied?.messages).includes('Brief source label'),
      'the short micro is what replaces the range in live context');
    const saved = after.spans[0];
    assert.equal(saved.summary, 'Full source summary for resumption');
    assert.equal(saved.micro, 'Brief source label');
    assert.equal(saved.startTurn, 1);
    assert.equal(saved.endTurn, 4);
    assert.match(saved.id, /^sp-[0-9a-f]{8}$/);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('malformed JSON consumes one attempt and a valid second completion is applied', async () => {
  let calls = 0;
  const { sandbox, home, hooks, ctx } = await setup('background-retry', async () => {
    calls++;
    return calls === 1
      ? { stopReason: 'stop', content: [{ type: 'text', text: 'not json' }] }
      : result('short', 'second attempt summary');
  });
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => calls);
    await tick();
    assert.equal(calls, 2, 'invalid output is retried once, never more');
    const applied = await hooks.get('context')({ messages }, ctx);
    const after = persistedState(sandbox);
    assert.ok(JSON.stringify(applied?.messages).includes('short'));
    assert.equal(after.spans[0].summary, 'second attempt summary');
    assert.equal(after.spans.length, 1);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('two background failures defer the old hook fallback until the next trigger', async () => {
  let calls = 0;
  const { sandbox, home, hooks, ctx } = await setup('background-fallback', async () => {
    calls++;
    throw new Error('provider unavailable');
  });
  const messages = conversation();
  try {
    const first = await reachBackground(hooks, ctx, messages, () => calls);
    assert.equal(hasGate(first), false, 'the failing trigger is not also the fallback trigger');
    await tick();
    assert.equal(calls, 2);
    assert.equal(persistedState(sandbox).backgroundSummary.status, 'exhausted');

    let fallback = null;
    for (let turn = 0; turn < 4 && !hasGate(fallback); turn++) {
      fallback = await hooks.get('context')({ messages }, ctx);
      if (!hasGate(fallback)) await hooks.get('turn_end')({}, ctx);
    }
    assert.equal(hasGate(fallback), true, 'the existing gate demand resumes on a later trigger');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a ready result for a range that moved is discarded, never applied to other text', async () => {
  let calls = 0;
  let resolveCompletion;
  const { sandbox, home, hooks, ctx } = await setup('background-stale-range', () => {
    calls++;
    return new Promise((resolve) => { resolveCompletion = resolve; });
  });
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => calls);
    const summarized = persistedState(sandbox).backgroundSummary;
    resolveCompletion(result('moved micro', 'moved summary'));
    await tick();

    // A new turn arrives before the ready result can be applied: the compressible
    // range is no longer the one that was summarized, so applying it would describe
    // the WRONG messages.
    const longer = [
      ...messages,
      { role: 'user', content: 'later question ' + 'Q'.repeat(220), timestamp: 900 },
      { role: 'assistant', content: 'later answer ' + 'B'.repeat(220), timestamp: 901 },
    ];
    await hooks.get('context')({ messages: longer }, ctx);
    const after = persistedState(sandbox);
    assert.equal(after.spans.length, 0, 'a moved range must not be compressed with a stale summary');
    assert.doesNotMatch(JSON.stringify(after.spans), /moved summary/);
    assert.notEqual(after.backgroundSummary?.requestId, summarized.requestId,
      'the moved range needs a new request, not the result of the old one');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a ready result whose range disappeared is marked stale and compresses nothing', async () => {
  let calls = 0;
  let resolveCompletion;
  const { sandbox, home, hooks, ctx } = await setup('background-stale-gone', () => {
    calls++;
    return new Promise((resolve) => { resolveCompletion = resolve; });
  });
  try {
    await reachBackground(hooks, ctx, conversation(), () => calls);
    resolveCompletion(result('orphan micro', 'orphan summary'));
    await tick();
    // The list no longer holds anything compressible: there is nothing to summarize.
    await hooks.get('context')({ messages: conversation().slice(0, 2) }, ctx);
    const after = persistedState(sandbox);
    assert.equal(after.backgroundSummary?.status, 'stale');
    assert.match(String(after.backgroundSummary?.lastError), /changed/);
    assert.equal(after.spans.length, 0);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('the manual tool and the background job share one anti-duplicate guard', async () => {
  let calls = 0;
  let resolveCompletion;
  const { sandbox, home, tools, hooks, ctx } = await setup('background-guard', () => {
    calls++;
    return new Promise((resolve) => { resolveCompletion = resolve; });
  });
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => calls);
    const manual = await tools.get('cwl_compress_range').execute(
      't', { summary: 'manual summary of the same range', micro: 'manual micro' }, undefined, undefined, ctx,
    );
    assert.equal(manual.details.ok, true, `the manual compression was refused: ${JSON.stringify(manual.details)}`);
    assert.equal(persistedState(sandbox).spans.length, 1);

    // The completion for the SAME range lands after the manual answer: it must not
    // become a second leaf for the same messages.
    resolveCompletion(result('late micro', 'late summary'));
    await tick();
    await hooks.get('context')({ messages }, ctx);
    const after = persistedState(sandbox);
    assert.equal(after.spans.length, 1, 'one range must never become two leaves');
    assert.equal(after.backgroundSummary, null, 'the manual commit spends the pending request');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a restart keeps the attempt count: the limit of two survives the process', async () => {
  let calls = 0;
  const { sandbox, home, hooks, ctx } = await setup('background-restart', async () => {
    calls++;
    throw new Error('provider unavailable');
  });
  const messages = conversation();
  const sessionFile = path.join(sandbox.dir, 'session.jsonl');
  try {
    await reachBackground(hooks, ctx, messages, () => calls);
    await tick();
    assert.equal(calls, 2, 'both attempts happen inside the first process');
    assert.equal(persistedState(sandbox).backgroundSummary?.status, 'exhausted');

    // A restart: same session file, fresh extension instance, empty in-flight map.
    const second = await bootExtension(sandbox, { name: 'restarted' });
    const ctx2 = sessionCtx(sessionFile);
    ctx2.model = ctx.model;
    ctx2.modelRegistry = { ...ctx.modelRegistry, complete: async () => { calls++; throw new Error('still unavailable'); } };
    await second.hooks.get('session_start')({}, ctx2);
    let last = null;
    for (let turn = 0; turn < 3; turn++) {
      last = await second.hooks.get('context')({ messages }, ctx2);
      await second.hooks.get('turn_end')({}, ctx2);
    }
    assert.equal(calls, 2, 'a restart must not grant a third completion for the same range');
    assert.equal(hasGate(last), true, 'after the restart the existing hook fallback resumes');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});
