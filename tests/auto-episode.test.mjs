import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// The budget is deliberately FAR out of reach. The automatic recording must not depend on being
// over budget: a graph that only fills once the context is already too big records ONE episode
// over the whole history, which is the opposite of a recording every N turns.
const config = {
  tokenBudget: 100_000,
  thresholdRatio: 1,
  protectedTurns: 2,
  autoDelimiterTurns: 2,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
};

const filler = (label) => `${label} ` + 'x'.repeat(140);

/**
 * A conversation where every turn is one user prompt, one assistant answer and one tool result,
 * which is the shape the whole episode mechanism is anchored on (`toolCallId` + `toolName`).
 */
const conversation = ({ turns = 6, toolName = 'bash', lastTurnTool = null, withTools = true } = {}) => {
  const messages = [];
  for (let t = 1; t <= turns; t++) {
    messages.push({ role: 'user', content: filler(`question ${t}`), timestamp: t * 10 });
    messages.push({ role: 'assistant', content: filler(`working ${t}`), timestamp: t * 10 + 1 });
    if (!withTools) continue;
    const name = lastTurnTool && t === turns ? lastTurnTool : toolName;
    messages.push({
      role: 'toolResult',
      toolCallId: `call-${t}`,
      toolName: name,
      content: [{ type: 'text', text: filler(`output ${t}`) }],
      timestamp: t * 10 + 2,
    });
  }
  return messages;
};

async function setup(name, overrides = {}) {
  const sandbox = makeSandbox({ name, config: { ...config, ...overrides } });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  ctx.model = { provider: 'test-provider', id: 'test-model' };
  ctx.modelRegistry = {
    find: (provider, id) => (provider === 'test-provider' && id === 'test-model' ? ctx.model : null),
    hasConfiguredAuth: () => true,
    // Reached only if the budget trigger fires, which this file keeps out of reach: a test that
    // silently summarised would be measuring something else.
    complete: () => { throw new Error('the background summarizer must not run in the auto-episode tests'); },
  };
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Drives `turns` turns, handing the hook one more turn of messages each time. */
async function drive(hooks, ctx, messages, turns) {
  let seen = 0;
  for (let round = 0; round < turns; round++) {
    seen = Math.min(messages.length, seen + 3);
    await hooks.get('context')({ messages: messages.slice(0, seen) }, ctx);
    await hooks.get('turn_end')({}, ctx);
  }
}

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const stateOf = (sandbox) => {
  const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.existsSync(stateDir)
    ? fs.readdirSync(stateDir).find((name) => name.endsWith('.json'))
    : undefined;
  return file ? JSON.parse(fs.readFileSync(path.join(stateDir, file), 'utf8')) : null;
};

/** Episodes as the tool sees them; `[]` when nothing was ever persisted. */
const episodesOf = (sandbox) => stateOf(sandbox)?.graph?.episodes ?? [];

test('an episode is recorded every N turns, anchored on the last tool result, and it is evictable', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-basic');
  await drive(hooks, ctx, conversation({ turns: 7 }), 7);

  const episodes = episodesOf(sandbox);
  assert.deepEqual(
    episodes.map((ep) => ep.name),
    ['auto-0', 'auto-2', 'auto-4', 'auto-6'],
    'one recording every 2 turns, named after the turn it was taken at',
  );
  for (const ep of episodes) {
    assert.equal(ep.level, 'none', 'a recorded episode must be a candidate for eviction');
    assert.equal(
      ep.type,
      'expl',
      'bash is not in MUTATING_TOOLS: which of its 236 calls wrote is not something a name can say',
    );
    assert.deepEqual(ep.dependencies, []);
    assert.ok(
      ep.endToolCallId && /^call-\d+$/.test(ep.endToolCallId),
      `the closing anchor must be a real tool result, got ${ep.endToolCallId}`,
    );
    assert.match(ep.description, /^auto: \d+ msgs, tools: bashx\d+$/);
  }
  assert.match(logOf(sandbox), /AUTO-EPISODE expl "auto-0"/);
  assert.doesNotMatch(logOf(sandbox), /BACKGROUND/, 'the budget machinery stayed out of the way');
});

test('each episode starts where the previous one closed, so the ranges tile the session', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-tiling');
  await drive(hooks, ctx, conversation({ turns: 7 }), 7);

  const episodes = episodesOf(sandbox);
  assert.equal(episodes.length, 4);
  assert.equal(episodes[0].startToolCallId, '', 'the first episode starts AT the session');
  for (let i = 1; i < episodes.length; i++) {
    assert.equal(
      episodes[i].startToolCallId,
      episodes[i - 1].endToolCallId,
      'the closing anchor of one episode is the opening anchor of the next',
    );
  }
  // The ones that closed before the last turn cannot share the same end: a shared end would mean
  // the recording fired without any new material in between.
  const ends = new Set(episodes.map((ep) => ep.endToolCallId));
  assert.equal(ends.size, episodes.length, 'every recording must have claimed NEW material');
});

test('a segment that wrote is an act', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-act');
  await drive(hooks, ctx, conversation({ turns: 5, toolName: 'edit' }), 5);

  const episodes = episodesOf(sandbox);
  assert.ok(episodes.length > 0, 'something must have been recorded');
  for (const ep of episodes) assert.equal(ep.type, 'act');
  assert.match(logOf(sandbox), /AUTO-EPISODE act "auto-0"/);
});

test('a `delimiter` result is never the closing anchor: it is a boundary, not work', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-delimiter', { autoDelimiterTurns: 1 });
  // The last turn ends with the agent's own delimiter, which claims the segment itself.
  await drive(hooks, ctx, conversation({ turns: 3, lastTurnTool: 'delimiter' }), 3);

  const episodes = episodesOf(sandbox);
  assert.deepEqual(
    episodes.map((ep) => ep.name),
    ['auto-0', 'auto-1'],
    'anchoring on the delimiter result would have produced a third episode out of nothing',
  );
  assert.ok(
    episodes.every((ep) => ep.endToolCallId !== 'call-3'),
    'the delimiter result must never be used as the closing anchor',
  );
});

test('no tool result means no episode: the anchor must exist in the list', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-no-anchor', { autoDelimiterTurns: 1 });
  await drive(hooks, ctx, conversation({ turns: 4, withTools: false }), 4);

  assert.deepEqual(episodesOf(sandbox), []);
  assert.doesNotMatch(logOf(sandbox), /AUTO-EPISODE/);
});

test('`autoDelimiterTurns: 0` disables the recording, and a bad value falls back on 3', async () => {
  const off = await setup('auto-ep-off', { autoDelimiterTurns: 0 });
  await drive(off.hooks, off.ctx, conversation({ turns: 4 }), 4);
  assert.deepEqual(episodesOf(off.sandbox), [], '0 is a legal value and it means OFF');
  assert.doesNotMatch(logOf(off.sandbox), /AUTO-EPISODE/);

  const bad = await setup('auto-ep-bad-value', { autoDelimiterTurns: -5 });
  await drive(bad.hooks, bad.ctx, conversation({ turns: 7 }), 7);
  assert.deepEqual(
    episodesOf(bad.sandbox).map((ep) => ep.name),
    ['auto-0', 'auto-3', 'auto-6'],
    'an out-of-range value falls back on the default of 3 instead of disabling the feature',
  );
});

test('one turn records one episode, however many hooks it runs, and the cadence is persisted', async () => {
  const { sandbox, hooks, ctx } = await setup('auto-ep-one-per-turn', { autoDelimiterTurns: 1 });
  const messages = conversation({ turns: 2 });

  // Three hooks with `turn_end` never called: it is one turn, and a tool loop really does run the
  // hook many times inside it.
  await hooks.get('context')({ messages: messages.slice(0, 3) }, ctx);
  await hooks.get('context')({ messages: messages.slice(0, 3) }, ctx);
  await hooks.get('context')({ messages: messages.slice(0, 3) }, ctx);

  assert.deepEqual(episodesOf(sandbox).map((ep) => ep.name), ['auto-0']);
  assert.equal((logOf(sandbox).match(/AUTO-EPISODE/g) ?? []).length, 1);
  assert.equal(
    stateOf(sandbox).lastAutoEpisodeTurn,
    0,
    'the turn of the last recording is PERSISTED, so a restart does not re-ask for it',
  );
});
