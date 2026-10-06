/**
 * ONE LEAF, ONE NODE.
 *
 * A leaf is the unit the head pays for: its micro is injected once, and the node that holds it
 * is the thing that can give it a name. If two nodes hold the same leaf, the leaf's text is
 * paid for twice, `cwl_group` refuses it as "already in the pit" while the buffer still keeps
 * it, and the two lists contradict each other.
 *
 * This suite pins the three fixes for that defect:
 *  1. `cwl_group` used to COPY the leaves that stayed behind into fresh chunk nodes without ever
 *     removing them from the node they came from. At the frontier the nodes involved are now
 *     EMPTIED, because the chunks REPLACE them.
 *  2. `refreshNodes` now enforces the invariant itself — one leaf, one node, FIRST wins in
 *     chronological order — so a state written by the old code HEALS on the next pass instead
 *     of staying wrong forever.
 *  3. `topicView` did not filter the pit, so every topic INSIDE the archive injected its own
 *     `TOPIC "name"` block on top of the pit block. It is a distinct, pre-existing defect.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  // Six loose leaves, THREE per node: the fixture must produce at least one ordinary node with
  // leaves to spare, because the defect lived in the leaves that STAY BEHIND when a group takes
  // some out. With two per node there is never a leftover to re-partition and the chunk path is
  // unreachable — the guard in the first test says so out loud instead of passing vacuously.
  looseLeaves: 6,
  nodeCapacity: 3,
  mergeNodesAt: 1,
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot() {
  const sandbox = makeSandbox({ name: `leaf-ownership-${seq++}`, config: config() });
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

const stateFileOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return path.join(dir, file);
};

const stateOf = (sandbox) => JSON.parse(fs.readFileSync(stateFileOf(sandbox), 'utf8'));

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const injected = (out) =>
  out
    .filter((m) => m.role === 'custom' && m.customType === 'cwl-compressed')
    .map((m) => String(m.content ?? ''))
    .join('\n---\n');

async function makeLeaves(sandbox, tools, hooks, ctx, n) {
  for (let i = 1; i <= n; i++) {
    await hook(hooks, ctx, conversation(1, i + 3));
    const res = await tools.get('cwl_compress_range').execute(
      't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
    );
    assert.equal(res.details.ok, true, `leaf ${i} was not born: ${JSON.stringify(res.details)}`);
  }
  const leaves = stateOf(sandbox).spans;
  for (let i = 0; i < leaves.length; i++) {
    const r = await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1} of the story` }, undefined, undefined, ctx);
    assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
  }
  await hook(hooks, ctx, conversation(1, 12));
  return stateOf(sandbox).spans.map((s) => s.id);
}

/** Every leaf id that more than one node claims, with the ids of the nodes that claim it. */
const doubleOwners = (st) => {
  const owners = new Map();
  for (const nd of st.nodes) {
    for (const leaf of nd.leaves) {
      if (!owners.has(leaf)) owners.set(leaf, []);
      owners.get(leaf).push(nd.id);
    }
  }
  return [...owners.entries()].filter(([, ids]) => ids.length > 1);
};

test('a group at the frontier never leaves a leaf in two nodes', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 10);
    const before = stateOf(sandbox);
    assert.deepEqual(doubleOwners(before), [], 'the fixture itself started with a double owner');

    // A source with leaves to SPARE. The defect was in the leaves that stay behind: they were
    // copied into fresh chunks without ever leaving the node they came from, so a source with
    // fewer than three leaves cannot reach that path at all — and this guard is what keeps the
    // test from passing for the wrong reason. An ordinary node (no description) is a legal
    // source: `groupBeyondBuffer` is on, and its leaves are still in the head as micros.
    const source = before.nodes.find((nd) => !nd.description && nd.leaves.length >= 3);
    assert.ok(
      source,
      `no ordinary node holds 3+ leaves, so the chunk path is unreachable: ${before.nodes
        .map((nd) => `${nd.id}:${nd.leaves.length}${nd.description ? '(topic)' : ''}`)
        .join(', ')}`,
    );

    // Two leaves are taken OUT, so the rest STAY BEHIND and must become chunks.
    const moving = source.leaves.slice(0, 2);
    const grouped = await tools.get('cwl_group').execute(
      't', { leaves: moving, name: 'gamma', description: 'GAMMA covers the two stories that were taken out' },
      undefined, undefined, ctx,
    );
    assert.equal(grouped.details.ok, true, `the group was refused: ${JSON.stringify(grouped.details)}`);

    const after = stateOf(sandbox);
    // THE INVARIANT: nothing is claimed twice. Before the fix the leaves that stayed behind were
    // copied into the chunk nodes AND left in the buffer, so this list was not empty.
    assert.deepEqual(doubleOwners(after), [], 'a leaf is held by two nodes at once');

    // And the leaves that stayed behind are still held — by the chunk, not by the emptied buffer.
    const stillHeld = new Set(after.nodes.flatMap((nd) => nd.leaves));
    for (const leaf of source.leaves.slice(2)) {
      assert.ok(stillHeld.has(leaf), `the leaf ${leaf} was dropped instead of moved into a chunk`);
    }
    const gamma = after.nodes.find((nd) => nd.id === grouped.details.id);
    assert.ok(gamma, 'the topic was not created');
    assert.deepEqual(gamma.leaves.slice().sort(), moving.slice().sort(), 'the topic did not take the moving leaves');
  } finally {
    home.restore(); sandbox.cleanup();
  }
});

test('a state with a leaf claimed twice heals on the next pass, and the FIRST node keeps it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 10);

    // A state as the OLD code would have written it: the same leaf inside two nodes, the second
    // one later in the index. This is not reachable through the tools any more — which is the
    // point of the fix — so it is written to disk BY HAND, exactly as it was measured in
    // production (`nd-69506c4e` and the buffer holding the same ten leaves).
    const file = stateFileOf(sandbox);
    const st = JSON.parse(fs.readFileSync(file, 'utf8'));
    const first = st.nodes[0];
    const last = st.nodes[st.nodes.length - 1];
    assert.notEqual(first.id, last.id, 'the fixture produced a single node');
    const stolen = first.leaves[0];
    assert.ok(stolen, 'the first node holds no leaf to steal');
    if (!last.leaves.includes(stolen)) last.leaves.push(stolen);
    fs.writeFileSync(file, JSON.stringify(st));
    assert.deepEqual(
      doubleOwners(JSON.parse(fs.readFileSync(file, 'utf8'))).length,
      1,
      'the corrupted state was not written',
    );

    // A RESTART: a second instance of the extension, in a file of its own so the module cache
    // does not hand back the instance that already has the state in memory. It reads the state
    // from disk, the corrupted one.
    const second = await bootExtension(sandbox, { name: 'ext2' });
    await second.hooks.get('session_start')({}, ctx);
    // The pass that repairs is the CONTEXT HOOK's: it is the one that calls `refreshNodes` on
    // every turn in production. A read-only tool would show the corruption exactly as it is —
    // which is worth knowing, and is why the repair belongs to the pass that already runs.
    await hook(second.hooks, ctx, conversation(1, 12));

    const healed = stateOf(sandbox);
    assert.deepEqual(doubleOwners(healed), [], 'the state was not repaired: a leaf is still claimed twice');
    const keeper = healed.nodes.find((nd) => nd.leaves.includes(stolen));
    assert.ok(keeper, 'the stolen leaf disappeared entirely');
    assert.equal(keeper.id, first.id, 'the leaf was kept by the WRONG node: the first in the index must win');
  } finally {
    home.restore(); sandbox.cleanup();
  }
});

test('a topic inside the pit does not inject its own block on top of the pit block', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 10);

    // A topic born at the frontier, with a description of its own.
    const alpha = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(4, 6), name: 'alpha', description: 'ALPHA covers the first stories' },
      undefined, undefined, ctx,
    );
    assert.equal(alpha.details.ok, true, `alpha was not born: ${JSON.stringify(alpha.details)}`);
    const withAlpha = injected(await hook(hooks, ctx, conversation(1, 12)));
    assert.match(withAlpha, /TOPIC "alpha"/, 'alpha is not in the head before the merge');

    // The merge absorbs the young nodes: alpha ends up INSIDE the pit.
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the first stories' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const st = stateOf(sandbox);
    assert.ok((st.oldNode.nodes ?? []).includes(alpha.details.id), 'alpha did not end up in the pit');

    // THE FIX: the pit block stands for alpha, and alpha must NOT inject a block of its own.
    // Before the fix `topicView` did not filter the pit, so the description was paid for twice.
    //
    // AND THE TURN MATTERS. A topic's block is PREPENDED only when NONE of its leaves is still
    // resolvable in the messages (`if (tv.node.leaves.some((id) => resolvedIds.has(id))) continue`).
    // While the old messages are still in the list the block is injected IN PLACE, and there the
    // `pit` branch of the message loop wins over the `topic` branch — so the pit would hide the
    // defect by accident and the test would pass without the fix. A NEW turn is the case that
    // discriminates: the old leaves are no longer in the messages, the block is prepended, and
    // the pit is the only thing allowed to speak for them.
    const after = injected(await hook(hooks, ctx, conversation(20, 25)));
    assert.match(after, /CWL OLD NODE/, 'the pit block is not injected');
    assert.doesNotMatch(after, /TOPIC "alpha"/, 'a topic INSIDE the pit is still injected on top of the pit block');

    // The description is not lost: the pit page still renders it.
    const page = await tools.get('cwl_open').execute('t', { id: st.oldNode.id }, undefined, undefined, ctx);
    assert.equal(page.details.ok, true, `opening the pit failed: ${JSON.stringify(page.details)}`);
    assert.match(String(page.content[0].text), new RegExp(alpha.details.id), 'the pit page lost the topic it holds');
  } finally {
    home.restore(); sandbox.cleanup();
  }
});
