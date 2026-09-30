/**
 * A topic can CONTAIN other topics.
 *
 * The rule that makes it safe is that containment only goes BACKWARD in time: a node absorbs
 * only nodes that come AFTER it in `st.nodes`. The parent's position, and everything before it
 * in the index, therefore never move — and since absorbing a LATER node removes content after
 * the parent's block anyway, appending a shape line to that block costs no cache of its own.
 * The head still pays ONE block per node: a parent injects its description, its children
 * inject NOTHING, and depth can be unlimited exactly because depth is never injected.
 *
 * The four directions in which this suite must die:
 *  1. the child's block stays in the head -> nesting saves nothing;
 *  2. the shape line is missing or wrong -> the parent's block does not say what it holds;
 *  3. the buffer (or an older node) is absorbed -> the index prefix moves, which is the one
 *     cost this design refuses to pay;
 *  4. a leaf of a CONTAINED node that loses its micro is dropped or re-formed -> the closure of
 *     `children` is not transitive, and the archive's material walks back into the head.
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
  // Six loose leaves: a loose leaf has no owner, so `cwl_group` may move it — which is how two
  // topics are born at the frontier in the same turn.
  looseLeaves: 6,
  nodeCapacity: 2,
  mergeNodesAt: 1,
  // The size guards are measured against labels of ~1.200 characters and this fixture writes
  // labels like `BLOCK-1`: lowered on purpose, the guard has its own test.
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot() {
  const sandbox = makeSandbox({ name: `nested-topics-${seq++}`, config: config() });
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

/** Every block the extension INJECTS into the context, as one string. */
const injected = (out) =>
  out
    .filter((m) => m.role === 'custom' && m.customType === 'cwl-compressed')
    .map((m) => String(m.content ?? ''))
    .join('\n---\n');

/** Creates `n` leaves, each with its micro, and returns their ids in creation order. */
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

test('a topic absorbs the topic after it, and the child block leaves the head', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 10);
    // Two topics at the frontier, from loose leaves: they end up CONSECUTIVE in the index.
    const alpha = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(4, 6), name: 'alpha', description: 'ALPHA covers the first stories' }, undefined, undefined, ctx,
    );
    assert.equal(alpha.details.ok, true, `alpha was not born: ${JSON.stringify(alpha.details)}`);
    const beta = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(6, 8), name: 'beta', description: 'BETA covers the second stories' }, undefined, undefined, ctx,
    );
    assert.equal(beta.details.ok, true, `beta was not born: ${JSON.stringify(beta.details)}`);

    const before = injected(await hook(hooks, ctx, conversation(1, 12)));
    assert.match(before, /TOPIC "alpha"/, 'alpha was not in the head before the absorption');
    assert.match(before, /TOPIC "beta"/, 'beta was not in the head before the absorption');

    // The absorption: alpha CONTAINS beta. No description is asked for, and none is rewritten.
    const absorbed = await tools.get('cwl_group').execute(
      't', { node: alpha.details.id, nodes: [beta.details.id] }, undefined, undefined, ctx,
    );
    assert.equal(absorbed.details.ok, true, `the absorption was refused: ${JSON.stringify(absorbed.details)}`);
    assert.deepEqual(absorbed.details.absorbed, [beta.details.id], 'the wrong node was absorbed');

    const st = stateOf(sandbox);
    const alphaNode = st.nodes.find((nd) => nd.id === alpha.details.id);
    assert.ok(alphaNode, 'alpha is gone from the state');
    assert.deepEqual(alphaNode.children, [beta.details.id], 'alpha does not hold beta');
    assert.equal(
      alphaNode.description,
      'ALPHA covers the first stories',
      'the description of the parent was rewritten: absorbing must never ask for a new text',
    );
    const betaNode = st.nodes.find((nd) => nd.id === beta.details.id);
    assert.ok(betaNode, 'the child was deleted: it must stay in the state and stay readable');
    assert.equal(betaNode.leaves.length, 2, 'the child lost its own leaves');

    // The head: ONE block for the parent, and the child's block is gone. The shape line is
    // measured by the code, so it cannot claim a number that is not true.
    const after = injected(await hook(hooks, ctx, conversation(1, 12)));
    assert.match(after, /TOPIC "alpha"/, 'the parent block left the head');
    assert.doesNotMatch(after, /TOPIC "beta"/, 'the child block is still in the head: nesting saves nothing');
    assert.match(after, /holds 1 node\(s\), 2 leaf\/leaves/, 'the shape line is missing or wrong');

    // And the child is still readable, whole.
    const page = await tools.get('cwl_open').execute('t', { id: beta.details.id }, undefined, undefined, ctx);
    assert.equal(page.details.ok, true, `the child page failed: ${JSON.stringify(page.details)}`);
    assert.match(String(page.content[0].text), /MICRO-7/, 'the child page lost the micros of its leaves');
  } finally {

    home.restore(); sandbox.cleanup();
  }
});

test('the containment refuses the buffer, an older node and itself', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 10);
    const alpha = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(4, 6), name: 'alpha', description: 'ALPHA covers the first stories' }, undefined, undefined, ctx,
    );
    assert.equal(alpha.details.ok, true, `alpha was not born: ${JSON.stringify(alpha.details)}`);
    const beta = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(6, 8), name: 'beta', description: 'BETA covers the second stories' }, undefined, undefined, ctx,
    );
    assert.equal(beta.details.ok, true, `beta was not born: ${JSON.stringify(beta.details)}`);
    const st = stateOf(sandbox);
    const buffer = st.nodes[st.nodes.length - 1];

    const onBuffer = await tools.get('cwl_group').execute(
      't', { node: alpha.details.id, nodes: [buffer.id] }, undefined, undefined, ctx,
    );
    assert.equal(onBuffer.details.ok, false, 'the buffer was absorbed: a topic cannot act as a buffer');
    assert.equal(onBuffer.details.why, 'node-is-the-buffer');

    // beta is YOUNGER than alpha: absorbing alpha into beta would move the index prefix.
    const forward = await tools.get('cwl_group').execute(
      't', { node: beta.details.id, nodes: [alpha.details.id] }, undefined, undefined, ctx,
    );
    assert.equal(forward.details.ok, false, 'an older node was absorbed: the index prefix moves');
    assert.equal(forward.details.why, 'only-backward');

    const self = await tools.get('cwl_group').execute(
      't', { node: alpha.details.id, nodes: [alpha.details.id] }, undefined, undefined, ctx,
    );
    assert.equal(self.details.ok, false, 'a node absorbed itself');
    assert.equal(self.details.why, 'node-contains-itself');

    // At the frontier a NEW parent over nodes is refused: the parent must already exist to hold
    // its position, so the containment there is always `node: <parent>` + `nodes: [<child>]`.
    const bornFromNodes = await tools.get('cwl_group').execute(
      't', { nodes: [beta.details.id], name: 'gamma', description: 'GAMMA would hold beta' }, undefined, undefined, ctx,
    );
    assert.equal(bornFromNodes.details.ok, false, 'a new parent was born from nodes at the frontier');
    assert.equal(bornFromNodes.details.why, 'nodes-need-a-topic');
  } finally {

    home.restore(); sandbox.cleanup();
  }
});

test('inside the pit a new parent gathers the topics after it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 10);
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the first stories' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    assert.ok((stateOf(sandbox).oldNode.nodes ?? []).length > 0, 'the pit was born with no node inside');

    // Two topics at the frontier, both AFTER the pit: they are what the parent will gather.
    const alpha = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(4, 6), name: 'alpha', description: 'ALPHA covers the first stories' }, undefined, undefined, ctx,
    );
    assert.equal(alpha.details.ok, true, `alpha was not born: ${JSON.stringify(alpha.details)}`);
    const beta = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(6, 8), name: 'beta', description: 'BETA covers the second stories' }, undefined, undefined, ctx,
    );
    assert.equal(beta.details.ok, true, `beta was not born: ${JSON.stringify(beta.details)}`);

    // The parent is born INSIDE the pit, over the two young topics. No synthesis is written and
    // the pit's own synthesis is not touched: that is the whole saving of this operation.
    const feature = await tools.get('cwl_group').execute(
      't', {
        pit: true, nodes: [alpha.details.id, beta.details.id],
        name: 'feature', description: 'FEATURE gathers the rounds that built the same thing',
      }, undefined, undefined, ctx,
    );
    assert.equal(feature.details.ok, true, `the parent was refused: ${JSON.stringify(feature.details)}`);
    const st = stateOf(sandbox);
    assert.ok(
      (st.oldNode.nodes ?? []).includes(feature.details.id),
      'the parent was not catalogued in the pit',
    );
    const parent = st.nodes.find((nd) => nd.id === feature.details.id);
    assert.deepEqual(
      [...(parent.children ?? [])].sort(),
      [alpha.details.id, beta.details.id].sort(),
      'the parent does not hold both topics',
    );
    assert.equal(parent.leaves.length, 0, 'the parent should hold nodes, not leaves of its own');

    const out = injected(await hook(hooks, ctx, conversation(1, 12)));
    assert.match(out, /TOPIC "feature"/, 'the parent block is not in the head');
    assert.doesNotMatch(out, /TOPIC "alpha"/, 'a contained topic is still injected');
    assert.doesNotMatch(out, /TOPIC "beta"/, 'a contained topic is still injected');
    assert.match(out, /holds 2 node\(s\), 4 leaf\/leaves/, 'the shape line does not count the subtree');
  } finally {

    home.restore(); sandbox.cleanup();
  }
});

test('a leaf of a contained node survives losing its micro', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();

  try {
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 10);
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the first stories' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const alpha = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(4, 6), name: 'alpha', description: 'ALPHA covers the first stories' }, undefined, undefined, ctx,
    );
    assert.equal(alpha.details.ok, true, `alpha was not born: ${JSON.stringify(alpha.details)}`);
    const beta = await tools.get('cwl_group').execute(
      't', { leaves: ids.slice(6, 8), name: 'beta', description: 'BETA covers the second stories' }, undefined, undefined, ctx,
    );
    assert.equal(beta.details.ok, true, `beta was not born: ${JSON.stringify(beta.details)}`);
    const feature = await tools.get('cwl_group').execute(
      't', {
        pit: true, nodes: [alpha.details.id, beta.details.id],
        name: 'feature', description: 'FEATURE gathers the rounds that built the same thing',
      }, undefined, undefined, ctx,
    );
    assert.equal(feature.details.ok, true, `the parent was refused: ${JSON.stringify(feature.details)}`);

    // beta is a CHILD of a pit node. Its leaves are spoken for by the parent's description, so
    // a micro that disappears must not take the leaf away: that is what the closure of
    // `children` is for, and it is the one thing a non-transitive set would get wrong.
    const before = stateOf(sandbox).nodes.find((nd) => nd.id === beta.details.id).leaves;
    for (const id of before) {
      const r = await tools.get('cwl_micro').execute('t', { id, text: '' }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `the un-absorb of ${id} was refused: ${JSON.stringify(r.details)}`);
    }
    await hook(hooks, ctx, conversation(1, 12));
    const after = stateOf(sandbox).nodes.find((nd) => nd.id === beta.details.id);
    assert.ok(after, 'the contained node was dropped from the state');
    assert.deepEqual(after.leaves, before, 'the contained node lost leaves the parent still stands for');
    // The invariant that does not need the implementation's own definition of "contained": a
    // leaf belongs to AT MOST ONE node. If it came back as a young node, it is held twice —
    // and the material the parent already stands for is paid for a second time in the head.
    for (const id of before) {
      const holders = stateOf(sandbox).nodes.filter((nd) => nd.leaves.includes(id)).map((nd) => nd.id);
      assert.deepEqual(
        holders,
        [beta.details.id],
        `the leaf ${id} of the contained node is held by ${holders.join(', ') || 'nothing'}: it was re-formed`,
      );
    }
  } finally {

    home.restore(); sandbox.cleanup();
  }
});
