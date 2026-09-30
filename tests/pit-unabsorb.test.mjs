/**
 * A leaf of the PIT whose micro disappears must not empty its topic.
 *
 * The failure this test is about was measured in a real session, from the log: the pit was
 * born at 18:31 (`OLD old-bf71809b: absorbed 2 node(s), 60 leaf/leaves`), survived a restart
 * at 19:41, and by 20:01 it was gone — `NODES: 4 node(s) [nd-9ab72d0f:30, nd-1e15dd8c:30,
 * nd-af299f46:7, nd-bd2d1393:4]` — with its 67 leaves back among the YOUNG nodes and the
 * summary standing alone (`pit 0n/0f`). The operator's words: "il nodo old e' sparito e mi
 * sono trovato le foglie sparse che sono diventate nodi da sole".
 *
 * The cause was in `refreshNodes`: EVERY node dropped the leaves that no longer had a micro,
 * and the pit's nodes are not like the others. A young node stands for its leaves with their
 * micros, so a leaf that is un-absorbed (its body back in the context) must leave it. A node
 * of the PIT stands for its leaves with the merge SUMMARY: its micros are irrelevant there,
 * so a micro that disappears must not take the leaf away — and it must not empty the topic,
 * because an emptied node dies, its id leaves `st.oldNode.nodes`, and the leaves it held are
 * re-formed as young ones: the archive walking back into the head.
 *
 * THE THREE DIRECTIONS IN WHICH THE TEST MUST DIE:
 *  1. the topic loses the leaf whose micro was removed -> the archive forgets a leaf it still
 *     stands for, and the pit page starts lying about its shape;
 *  2. the emptied topic dies and its id leaves `st.oldNode.nodes` -> the pit is a summary
 *     with no nodes;
 *  3. the leaf is re-formed into a young node -> the material the pit already covers is paid
 *     for a second time in the head.
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
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 1,
  // The size guards of a merge are lowered on purpose: they are measured against labels of
  // ~1,200 characters and this fixture writes labels like `BLOCK-1`. The guard has its own test.
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot() {
  const sandbox = makeSandbox({ name: `pit-unabsorb-${seq++}`, config: config() });
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

test('a micro removed from a leaf of the pit does not empty its topic', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Six leaves, each with its micro: with looseLeaves 1 and nodeCapacity 2 they form nodes.
    for (let i = 1; i <= 6; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `leaf ${i} was not born: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox).spans;
    for (let i = 0; i < leaves.length; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
    }
    await hook(hooks, ctx, conversation(1, 12));

    // The merge: the pit is born with one node inside.
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the first stories' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const pit = merged.details.id;
    const beforeIds = [...(stateOf(sandbox).oldNode.nodes ?? [])];
    assert.ok(beforeIds.length > 0, 'the pit was born with no node inside: the test cannot measure what it is about');
    const pitNodeId = beforeIds[0];
    const victim = stateOf(sandbox).nodes.find((nd) => nd.id === pitNodeId).leaves[0];
    assert.ok(victim, 'the topic holds no leaf');

    // The un-absorb: the body goes back into the context and the micro disappears.
    const un = await tools.get('cwl_micro').execute('t', { id: victim, text: '' }, undefined, undefined, ctx);
    assert.equal(un.details.ok, true, `the un-absorb was refused: ${JSON.stringify(un.details)}`);

    await hook(hooks, ctx, conversation(1, 12));
    const after = stateOf(sandbox).oldNode;
    assert.ok(after, `the pit is gone: ${JSON.stringify(after)}`);
    assert.equal(after.id, pit, 'the pit changed identity');
    // The pit's node list is a CLAIM about what the archive holds: removing a micro from one
    // leaf of it must not change that claim by a single id.
    assert.equal(
      (after.nodes ?? []).join(','),
      beforeIds.join(','),
      `the pit's node list changed when a leaf of it lost its micro: ${beforeIds.join(',')} -> ${(after.nodes ?? []).join(',')}`,
    );
    const topic = stateOf(sandbox).nodes.find((nd) => nd.id === pitNodeId);
    assert.ok(topic, 'the topic node was dropped from the state');
    assert.ok(
      topic.leaves.includes(victim),
      'the un-absorbed leaf left its topic: the archive drops a leaf the synthesis still stands for',
    );
    const young = stateOf(sandbox).nodes.filter((nd) => !after.nodes.includes(nd.id));
    assert.ok(
      !young.some((nd) => nd.leaves.includes(victim)),
      'the un-absorbed leaf was re-formed into a young node: the material the pit covers is paid for twice',
    );

    // And the FULL symptom the log shows: a topic that loses ALL its leaves. Without the fix
    // the node empties, dies, its id leaves `st.oldNode.nodes` — the pit becomes a summary with
    // no nodes (`pit 0n/0f`) — and its leaves are re-formed as young ones.
    const pitLeaves = [...(stateOf(sandbox).nodes.find((nd) => nd.id === pitNodeId)?.leaves ?? [])];
    assert.ok(pitLeaves.length > 0, 'the topic holds no leaf: the full symptom cannot be measured');
    for (const id of pitLeaves) {
      const r = await tools.get('cwl_micro').execute('t', { id, text: '' }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `the un-absorb of ${id} was refused: ${JSON.stringify(r.details)}`);
    }
    await hook(hooks, ctx, conversation(1, 12));
    const emptied = stateOf(sandbox).oldNode;
    assert.ok(
      (emptied.nodes ?? []).includes(pitNodeId),
      `the topic left the pit when all its leaves lost their micro: the pit is now ${JSON.stringify(emptied.nodes)}`,
    );
    const kept = stateOf(sandbox).nodes.find((nd) => nd.id === pitNodeId);
    assert.equal(
      (kept?.leaves ?? []).length,
      pitLeaves.length,
      'the topic lost leaves the synthesis still stands for',
    );
    const youngNow = stateOf(sandbox).nodes.filter((nd) => !(emptied.nodes ?? []).includes(nd.id));
    for (const id of pitLeaves) {
      assert.ok(
        !youngNow.some((nd) => nd.leaves.includes(id)),
        `${id} came back as a young node: the archive pays twice for material the pit already covers`,
      );
    }
  } finally { home.restore(); sandbox.cleanup(); }
});
