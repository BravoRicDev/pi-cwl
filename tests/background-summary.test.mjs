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

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

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

test('two background failures do not disarm the channel: a NEW range gets its own attempts', async () => {
  let calls = 0;
  let fail = true;
  const { sandbox, home, hooks, ctx } = await setup('background-fallback', async () => {
    calls++;
    if (fail) throw new Error('provider unavailable');
    return result('recovered micro', 'recovered summary');
  });
  const messages = conversation();
  try {
    const first = await reachBackground(hooks, ctx, messages, () => calls);
    assert.equal(hasGate(first), false, 'the failing trigger is not also the fallback trigger');
    await tick();
    assert.equal(calls, 2);
    const after = persistedState(sandbox);
    assert.equal(after.backgroundSummary.status, 'exhausted');

    // THE LATCH IS GONE, and its absence is the contract under test. It used to disarm the
    // background summarizer after two failures and hand the job over to a `delimiter` demand
    // shown to the agent; that demand left the context by decision, so all the latch could still
    // do was stop the extension from scheduling anything at all, for ever. What replaced it is a
    // wait BOUNDED by `EXHAUSTED_RETRY_TURNS`, and the two assertions below measure both halves:
    // the provider is not hammered, and the channel is not disarmed.
    assert.equal(
      'backgroundFallbackAfterTurn' in after,
      false,
      'the exhausted request disarmed the background channel: the fallback it names no longer exists',
    );
    assert.match(
      logOf(sandbox),
      /BACKGROUND summary exhausted: .+ after 2 attempts: /,
      'the exhaustion was not reported together with the reason the provider gave',
    );

    // INSIDE the cooldown nothing is retried, even with new material in the list: the range's END
    // drifts on every turn, so "a new requestId" is not new material, and retrying would cost two
    // failed provider calls per turn during an outage. The old latch was the same idea with no end.
    messages.push({ role: 'user', content: 'more material ' + 'N'.repeat(220), timestamp: 100 });
    messages.push({ role: 'assistant', content: 'more answer ' + 'M'.repeat(220), timestamp: 101 });
    let last = null;
    for (let turn = 0; turn < 2; turn++) {
      last = await hooks.get('context')({ messages }, ctx);
      await hooks.get('turn_end')({}, ctx);
    }
    assert.equal(
      hasGate(last),
      false,
      'a demand was injected: the context is closed by the operator, the budget no longer writes into it',
    );
    assert.equal(calls, 2, 'the exhausted material was retried inside the cooldown: the provider would be hammered');

    // AND THE WAIT ENDS. This is the assertion that would have caught the dead latch: with
    // `fallbackActive` on the scheduling line, the extension went on refusing to schedule for the
    // rest of the session and the context stayed over budget for ever.
    fail = false;
    for (let turn = 0; turn < 20 && calls < 3; turn++) {
      await hooks.get('context')({ messages }, ctx);
      await hooks.get('turn_end')({}, ctx);
    }
    assert.equal(calls, 3, 'the cooldown never expired: the background channel is disarmed for good');
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
    assert.equal(calls, 2, 'a restart must not grant a third completion while the cooldown runs');
    assert.equal(
      hasGate(last),
      false,
      'a demand was injected after the restart: the fallback is a marker in the state, not a message in the context',
    );
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('an OpenCode background completion carries the session header the gateway requires', async () => {
  const seen = [];
  const { sandbox, home, hooks, ctx } = await setup('background-opencode-header', async (...args) => {
    seen.push(args[2]);
    return result('header micro', 'header summary');
  });
  // The gateway answers 400 MissingSessionID without this header, and the adapter
  // turns that into a bare 'error' stop reason: the sessionId option alone is not
  // enough, because the bare complete() path merges only the headers we pass.
  ctx.model = { provider: 'opencode-go', id: 'deepseek-v4.1-flash', baseUrl: 'https://opencode.ai/zen/go/v1' };
  ctx.sessionManager.getSessionId = () => 'sess-header-id';
  ctx.modelRegistry = {
    ...ctx.modelRegistry,
    find: (provider, id) => (provider === ctx.model.provider && id === ctx.model.id ? ctx.model : null),
  };
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => seen.length);
    assert.equal(seen.length, 1, 'the hook starts exactly one completion');
    assert.equal(seen[0].headers?.['x-opencode-session'], 'sess-header-id',
      'the request went out without the header the OpenCode gateway requires');
    assert.equal(seen[0].headers?.['x-opencode-client'], 'pi');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a non-OpenCode provider receives no session header at all', async () => {
  const seen = [];
  const { sandbox, home, hooks, ctx } = await setup('background-plain-provider', async (...args) => {
    seen.push(args[2]);
    return result('plain micro', 'plain summary');
  });
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => seen.length);
    assert.equal(seen.length, 1, 'the hook starts exactly one completion');
    assert.equal(seen[0].headers, undefined, 'the fix must leave every other provider byte-identical');
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});

test('a provider failure reports its own reason, not only the stop reason', async () => {
  let calls = 0;
  const reason = 'Request is missing x-opencode-session and cannot be routed efficiently.';
  const { sandbox, home, hooks, ctx } = await setup('background-error-detail', async () => {
    calls++;
    return { stopReason: 'error', errorMessage: reason, content: [] };
  });
  const messages = conversation();
  try {
    await reachBackground(hooks, ctx, messages, () => calls);
    for (let i = 0; i < 6 && calls < 2; i++) await tick();
    const after = persistedState(sandbox);
    assert.equal(calls, 2, 'a failed attempt is retried exactly once');
    assert.equal(after.backgroundSummary.status, 'exhausted');
    assert.match(String(after.backgroundSummary.lastError), /x-opencode-session/,
      'the provider reason was discarded: the log would say only completion stopped: error');
    assert.match(logOf(sandbox), /x-opencode-session/);
  } finally {
    await hooks.get('session_shutdown')?.({}, ctx);
    home.restore();
    sandbox.cleanup();
  }
});
