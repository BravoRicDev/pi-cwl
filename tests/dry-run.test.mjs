/**
 * DRY RUNS, STRUCTURED ERRORS, PAGINATION.
 *
 * Three additions that exist to make the agent spend FEWER calls, and the ways each of them
 * could be a lie:
 *
 *  1. a dry run that leaves a trace is worse than no dry run: the state is the LIVE object the
 *     next tool call reads, so if the prediction mutates it the agent acts on a world that has
 *     already moved. Every test here asserts the state file is byte-for-byte what it was.
 *  2. a dry run that only reports what it would APPROVE is half useful: the call worth
 *     predicting is the one that gets REFUSED, because that is the one that costs a second
 *     call today. So the refusal case is tested explicitly, with the numbers.
 *  3. a structured error that drops the fields it already had would break every caller that
 *     reads `error`/`why`/`detail`: the new keys are ADDITIVE and the old ones are asserted
 *     to still be there.
 *  4. pagination whose DEFAULT truncates would silently change what every existing caller
 *     sees. The default must return everything, and `total` must always be the real total.
 *  5. leaf ids in `cwl_status` must stay OFF by default: that tool is called often and the ids
 *     are the part of it that grows with the session, so a default that lists them makes every
 *     caller pay for a line it may never read.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (over = {}) => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  // Two loose leaves: a loose leaf has no owner, so `cwl_group` may move it. The OTHER four
  // have to end up inside nodes, otherwise there is no buffer to group against and no list
  // for `cwl_pending` to page — with `looseLeaves` at six the fixture produced NO node at all
  // and every call came back `no-buffer`.
  looseLeaves: 2,
  nodeCapacity: 2,
  mergeNodesAt: 1,
  // The size guards are measured against labels of ~1.200 characters; this fixture writes
  // labels like `BLOCK-1`, so they are lowered here and raised on purpose in their own test.
  mergeMinRatio: 0,
  mergeMinChars: 0,
  ...over,
});

async function boot(over) {
  const sandbox = makeSandbox({ name: `dry-run-${seq++}`, config: config(over) });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const stateOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const call = (tools, name, args, ctx) => tools.get(name).execute('t', args, undefined, undefined, ctx);

/** Creates `n` leaves, each with its micro, and returns their ids in creation order. */
async function makeLeaves(sandbox, tools, hooks, ctx, n) {
  for (let i = 1; i <= n; i++) {
    await hook(hooks, ctx, conversation(1, i + 3));
    const res = await call(tools, 'cwl_compress_range', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, ctx);
    assert.equal(res.details.ok, true, `leaf ${i} was not born: ${JSON.stringify(res.details)}`);
  }
  const leaves = stateOf(sandbox).spans;
  for (let i = 0; i < leaves.length; i++) {
    const r = await call(tools, 'cwl_micro', { id: leaves[i].id, text: `MICRO-${i + 1} of the story` }, ctx);
    assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
  }
  await hook(hooks, ctx, conversation(1, 12));
  return stateOf(sandbox).spans.map((s) => s.id);
}

// ---------------------------------------------------------------------------
// #2 — the dry run must not leave a trace
// ---------------------------------------------------------------------------

test('a dry run of cwl_group predicts the new topic and records nothing', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 4);
    const before = stateOf(sandbox);

    const dry = await call(tools, 'cwl_group', { leaves: ids, name: 'round-one', description: 'the first round', dryRun: true }, ctx);
    assert.equal(dry.details.ok, true, `the dry run failed: ${JSON.stringify(dry.details)}`);
    assert.equal(dry.details.dryRun, true, 'the answer does not say it was a dry run');
    assert.ok(String(dry.details.id).startsWith('nd-'), `no id was predicted: ${dry.details.id}`);
    assert.equal(dry.details.leaves, ids.length, 'the dry run miscounted the leaves');
    assert.equal(
      dry.details.microChars,
      stateOf(sandbox).spans.filter((s) => ids.includes(s.id)).reduce((n, s) => n + (s.micro ? s.micro.length : 0), 0),
      'the dry run miscounted the label characters',
    );
    assert.match(String(dry.content[0].text), /DRY RUN/, 'the answer is not marked as a dry run');

    // THE ASSERTION THAT MATTERS: the state on disk is exactly what it was. A dry run that
    // saved would make the next call read a world that has already moved.
    assert.deepEqual(stateOf(sandbox), before, 'the dry run wrote to the state');
    assert.equal(stateOf(sandbox).nodes.some((nd) => nd.id === dry.details.id), false, 'the predicted topic exists in the state');

    // And the REAL call does the same thing, on the same ids: the prediction was honest.
    const real = await call(tools, 'cwl_group', { leaves: ids, name: 'round-one', description: 'the first round' }, ctx);
    assert.equal(real.details.ok, true, `the real call failed: ${JSON.stringify(real.details)}`);
    assert.equal(real.details.id, dry.details.id, 'the real call produced a different id than the dry run predicted');
    assert.equal(stateOf(sandbox).nodes.some((nd) => nd.id === real.details.id), true, 'the real call created nothing');
  } finally {
    home.restore();
  }
});

test('a dry run of cwl_old predicts the merge and records nothing', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 6);
    const before = stateOf(sandbox);
    assert.equal(before.oldNode ?? null, null, 'the pit already existed before the merge');

    const dry = await call(tools, 'cwl_old', { text: 'SUMMARY of the first nodes', dryRun: true }, ctx);
    assert.equal(dry.details.ok, true, `the dry run failed: ${JSON.stringify(dry.details)}`);
    assert.equal(dry.details.dryRun, true, 'the answer does not say it was a dry run');
    assert.equal(dry.details.wouldProceed, true, 'the dry run says the merge would be refused');
    assert.equal(dry.details.destination, null, 'the dry run invented a pit');
    assert.ok(dry.details.nodes.length > 0, 'the dry run named no node to absorb');
    assert.ok(dry.details.leaves > 0, 'the dry run named no leaf');
    assert.ok(dry.details.needChars <= dry.details.freedChars, 'the dry run says it passes while the numbers say it does not');
    assert.deepEqual(stateOf(sandbox), before, 'the dry run wrote to the state');
    assert.equal(stateOf(sandbox).oldNode ?? null, null, 'the dry run created the pit');

    const real = await call(tools, 'cwl_old', { text: 'SUMMARY of the first nodes' }, ctx);
    assert.equal(real.details.ok, true, `the real call failed: ${JSON.stringify(real.details)}`);
    assert.ok(stateOf(sandbox).oldNode, 'the real call created no pit');
    assert.deepEqual(real.details.nodes, dry.details.nodes.length, 'the real call absorbed a different number of nodes');
  } finally {
    home.restore();
  }
});

test('a dry run predicts a REFUSAL, with the numbers that cause it', async () => {
  // The guard is raised out of reach: every merge must be refused, and that is exactly the
  // case worth predicting — it is the one that costs a second call today.
  const { sandbox, home, tools, hooks, ctx } = await boot({ mergeMinChars: 10_000_000 });
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 6);
    const before = stateOf(sandbox);

    const dry = await call(tools, 'cwl_old', { text: 'SUMMARY of the first nodes', dryRun: true }, ctx);
    assert.equal(dry.details.dryRun, true, 'the answer does not say it was a dry run');
    assert.equal(dry.details.wouldProceed, false, 'the dry run approved a merge the guard refuses');
    assert.equal(dry.details.needChars, 10_000_000, `the dry run reported the wrong floor: ${dry.details.needChars}`);
    assert.match(String(dry.content[0].text), /REFUSED/, 'the dry run does not say it would be refused');
    assert.deepEqual(stateOf(sandbox), before, 'the dry run wrote to the state');

    // The refusal itself, and the structured error that goes with it.
    const real = await call(tools, 'cwl_old', { text: 'SUMMARY of the first nodes' }, ctx);
    assert.equal(real.details.ok, false, 'the real call went through the raised guard');
    assert.equal(real.details.error, 'too-small', 'the old `error` field changed');
    assert.equal(real.details.code, 'synthesis-too-expensive', 'the structured code is missing');
    assert.equal(real.details.constraint.needChars, 10_000_000, 'the violated constraint is not reported');
    assert.equal(real.details.constraint.freedChars, dry.details.freedChars, 'the refusal and the prediction disagree on what would leave');
    assert.ok(Array.isArray(real.details.allowed) && real.details.allowed.length > 0, 'no way out is offered');
  } finally {
    home.restore();
  }
});

// ---------------------------------------------------------------------------
// #6 — structured errors are ADDITIVE
// ---------------------------------------------------------------------------

test('a refusal carries code, ids and allowed, and keeps why and detail', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 4);

    const bad = await call(tools, 'cwl_group', { leaves: ['sp-00000000'], name: 'x', description: 'y' }, ctx);
    assert.equal(bad.details.ok, false, 'an unknown leaf was accepted');
    assert.equal(bad.details.code, 'unknown-leaf', 'the structured code is missing');
    assert.deepEqual(bad.details.ids, ['sp-00000000'], 'the offending id is not in `ids`');
    assert.equal(bad.details.error, 'group-refused', 'the old `error` field changed');
    assert.equal(bad.details.why, 'unknown-leaf', 'the old `why` field changed');
    assert.equal(bad.details.detail, 'sp-00000000', 'the old `detail` field changed');
    assert.ok(bad.details.code === bad.details.why, 'the code and the prose disagree');

    // A topic too small to be worth its own description: the numbers, not just the sentence.
    const small = await call(tools, 'cwl_group', { leaves: [], name: 'x', description: 'y' }, ctx);
    assert.equal(small.details.ok, false, 'a topic with no leaves was accepted');
    assert.equal(small.details.code, 'no-leaves', 'the structured code is missing');
    assert.ok(Array.isArray(small.details.ids), '`ids` is missing, so a caller has to parse prose');
  } finally {
    home.restore();
  }
});

// ---------------------------------------------------------------------------
// #4 — pagination whose default does NOT truncate
// ---------------------------------------------------------------------------

test('cwl_pending pages only when asked, and always reports the real total', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 6);

    const all = await call(tools, 'cwl_pending', {}, ctx);
    assert.equal(all.details.ok, true, `cwl_pending failed: ${JSON.stringify(all.details)}`);
    assert.ok(all.details.total > 1, `the fixture produced no list to page: ${all.details.total}`);
    assert.equal(all.details.shown, all.details.total, 'the DEFAULT truncated the list');
    assert.equal(all.details.nextCursor, null, 'a complete page still offers a cursor');

    const first = await call(tools, 'cwl_pending', { limit: 1 }, ctx);
    assert.equal(first.details.shown, 1, 'limit was ignored');
    assert.equal(first.details.total, all.details.total, 'the total changed with the page size');
    assert.equal(first.details.nextCursor, 1, 'the cursor of the next page is wrong');
    assert.ok(first.content[0].text.length < all.content[0].text.length, 'the page is as long as the whole list');

    const second = await call(tools, 'cwl_pending', { limit: 1, cursor: first.details.nextCursor }, ctx);
    assert.equal(second.details.shown, 1, 'the second page is empty');
    assert.notDeepEqual(second.details.shownIds, first.details.shownIds, 'the second page repeats the first');
    // `ids` is the WHOLE list and stays that way: it is what every existing caller reads.
    assert.deepEqual(second.details.ids, first.details.ids, '`ids` stopped being the whole list: the default contract changed');
  } finally {
    home.restore();
  }
});

test('cwl_node pages its leaves only when asked', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 4);
    const grouped = await call(tools, 'cwl_group', { leaves: ids, name: 'round-one', description: 'the first round' }, ctx);
    const nodeId = grouped.details.id;

    const all = await call(tools, 'cwl_node', { id: nodeId }, ctx);
    assert.equal(all.details.ok, true, `cwl_node failed: ${JSON.stringify(all.details)}`);
    assert.equal(all.details.total, all.details.leafIds.length, 'the total is not the real number of leaves');
    assert.equal(all.details.nextCursor, null, 'a complete page still offers a cursor');

    const first = await call(tools, 'cwl_node', { id: nodeId, limit: 1 }, ctx);
    assert.equal(first.details.shown, 1, 'limit was ignored');
    assert.equal(first.details.total, all.details.total, 'the total changed with the page size');
    assert.deepEqual(first.details.leafIds, all.details.leafIds, '`leafIds` was truncated: the default contract changed');
    assert.deepEqual(first.details.shownIds, [all.details.leafIds[0]], 'the page is not the head of the list');
  } finally {
    home.restore();
  }
});

// ---------------------------------------------------------------------------
// #3 — cwl_pending says WHERE the leaves are
// ---------------------------------------------------------------------------

test('cwl_pending names the pit and says which operation applies to it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 6);

    const before = await call(tools, 'cwl_pending', {}, ctx);
    assert.deepEqual(before.details.pit, [], 'the pit section appeared with no pit');

    const merged = await call(tools, 'cwl_old', { text: 'SUMMARY of the first nodes' }, ctx);
    assert.equal(merged.details.ok, true, `the merge failed: ${JSON.stringify(merged.details)}`);

    const after = await call(tools, 'cwl_pending', {}, ctx);
    assert.ok(after.details.pit.length > 0, 'the pit is not named, so an agent cannot tell where its leaves are');
    assert.match(after.content[0].text, /pit: true/, 'the answer does not say which operation applies to the pit');
  } finally {
    home.restore();
  }
});

// ---------------------------------------------------------------------------
// #1 — the leaf ids in cwl_status are OPT-IN
// ---------------------------------------------------------------------------

test('cwl_status lists the leaf ids only when asked', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 6);

    const off = await call(tools, 'cwl_status', {}, ctx);
    assert.equal(/Leaf ids:/.test(off.content[0].text), false, 'the leaf ids appear by default: every caller pays for them');

    const on = await call(tools, 'cwl_status', { leafIds: true }, ctx);
    assert.match(on.content[0].text, /Leaf ids:/, 'the option did not add the leaf ids');
    assert.ok(ids.some((id) => on.content[0].text.includes(id)), 'no leaf id was listed, so the line is empty');
    assert.ok(on.content[0].text.length > off.content[0].text.length, 'the option added nothing');
  } finally {
    home.restore();
  }
});
