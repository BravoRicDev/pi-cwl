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

/** A genuinely later span, born after the archive is created. */
async function oneMoreLeaf(sandbox, hooks, ctx, tools, n) {
  const before = stateOf(sandbox).spans.length;
  await hook(hooks, ctx, conversation(1, n + 3));
  const res = await tools.get('cwl_compress_range').execute(
    't', { summary: `LATE-${n} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
  );
  assert.equal(res.details.ok, true, `late leaf was refused: ${JSON.stringify(res.details)}`);
  const spans = stateOf(sandbox).spans;
  assert.equal(spans.length, before + 1, `late leaf was not added: ${JSON.stringify(res.details)}`);
  const id = spans[spans.length - 1].id;
  const micro = await tools.get('cwl_micro').execute('t', { id, text: `LATE-MICRO-${n} of a later event` }, undefined, undefined, ctx);
  assert.equal(micro.details.ok, true, `late micro failed: ${JSON.stringify(micro.details)}`);
  await hook(hooks, ctx, conversation(1, n + 10));
  return id;
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
    const ids = await makeLeaves(sandbox, tools, hooks, ctx, 11);
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
    // This fixture's short synthesis wins over the longer internal descriptions, so a shape/count
    // accidentally added to the injected header would be the only changing part of this block.
    const beforePitBlock = (await hook(hooks, ctx, conversation(1, 12)))
      .find((m) => m.role === 'custom' && m.customType === 'cwl-compressed' && String(m.content).includes('CWL OLD NODE'));
    assert.ok(beforePitBlock, 'the pit block is not injected before cataloguing');
    assert.doesNotMatch(String(beforePitBlock.content), /what follows is NOT the merge synthesis/i);
    assert.doesNotMatch(String(beforePitBlock.content), /PIT SHAPE|older node\(s\) merged/i);

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

    const afterContext = await hook(hooks, ctx, conversation(1, 12));
    const out = injected(afterContext);
    // THE PIT WINS, and this is a DECISION, not an accident: a topic that lives INSIDE the pit
    // injects NOTHING of its own. The pit block stands for the whole archive — that is what a
    // synthesis IS — and paying a `topicHead` on top of it is paying twice for the same material.
    // MEASURED on a live session: the pit held 22 topics with a description and the head carried
    // the synthesis (2187 chars) PLUS 22 blocks, which cost more than the synthesis itself; and
    // `indexShape` never counted them, so the index line understated the head by more than half.
    // The names are not lost: they live in `cwl_open` on the pit page, which the block itself
    // tells the reader to call, and the page is checked below.
    assert.doesNotMatch(out, /TOPIC "feature"/, 'a topic inside the pit injected its own block on top of the pit block');
    assert.doesNotMatch(out, /TOPIC "alpha"/, 'a contained topic is still injected');
    assert.doesNotMatch(out, /TOPIC "beta"/, 'a contained topic is still injected');
    assert.doesNotMatch(out, /MICRO-7|MICRO-8/, 'a nested pit leaf is injected a second time outside the pit block');
    const afterPitBlock = afterContext.find((m) => m.role === 'custom' && m.customType === 'cwl-compressed' && String(m.content).includes('CWL OLD NODE'));
    assert.ok(afterPitBlock, 'the pit block disappeared after cataloguing');
    assert.equal(afterPitBlock.content, beforePitBlock.content, 'cataloguing changed the injected pit prefix although the synthesis did not change');

    // Consultation renders the full containment closure even though the injected block does
    // not pay for a changing outline. Its reported node count must describe that same closure,
    // not just the direct roots in oldNode.nodes.
    const pitBeforeOpen = stateOf(sandbox);
    const held = new Set(pitBeforeOpen.oldNode.nodes ?? []);
    let grew = true;
    while (grew) {
      grew = false;
      for (const nd of pitBeforeOpen.nodes) {
        if (!held.has(nd.id)) continue;
        for (const child of nd.children ?? []) {
          if (held.has(child)) continue;
          held.add(child);
          grew = true;
        }
      }
    }
    const expectedPitNodes = pitBeforeOpen.nodes.filter((nd) => held.has(nd.id)).length;
    const pitPage = await tools.get('cwl_open').execute('t', { id: st.oldNode.id }, undefined, undefined, ctx);
    assert.equal(pitPage.details.ok, true, `opening the pit failed: ${JSON.stringify(pitPage.details)}`);
    assert.equal(pitPage.details.nodes, expectedPitNodes, 'the pit page count omitted nested descendants');
    assert.ok(String(pitPage.content[0].text).includes(`${expectedPitNodes} node(s) inside`), 'the rendered pit heading disagrees with its node list');
    assert.match(String(pitPage.content[0].text), new RegExp(alpha.details.id));
    assert.match(String(pitPage.content[0].text), new RegExp(beta.details.id));

    // THE CONTAINER'S OWN ROW. `feature` holds 0 leaves of its own and 2 children, and the page
    // used to print `0 leaf/leaves (undefined .. undefined)` for it: `nd.leaves[0]` on an empty
    // array is `undefined`. A page that prints `undefined` is a page a reader cannot trust. The
    // row must carry what the container really holds — and this page is now the ONLY place that
    // says it, because the injected block no longer pays for a topic inside the pit.
    //
    // The finder targets the CATALOGUE row and not the outline: the outline labels a node by its
    // NAME (`├ feature [2 sub]`), so only this row carries the id and the word TOPIC.
    const pageText = String(pitPage.content[0].text);
    const containerRow = pageText.split('\n').find((line) => line.includes('TOPIC "feature"'));
    assert.ok(containerRow, `the container has no row on the pit page:\n${pageText}`);
    assert.match(containerRow, /holds 2 node\(s\), 4 leaf\/leaves/, 'the container row does not say what it holds');
    assert.doesNotMatch(containerRow, /undefined/, 'the container row printed an undefined endpoint');

    // A new root must be placed after the whole existing closure, not merely after the direct
    // pit roots. Give it a later frontier child to make that ordering observable.
    const laterLeaves = [];
    for (const n of [40, 50, 60]) laterLeaves.push(await oneMoreLeaf(sandbox, hooks, ctx, tools, n));
    const gamma = await tools.get('cwl_group').execute(
      't', { leaves: laterLeaves, name: 'gamma', description: 'GAMMA is a later frontier topic' }, undefined, undefined, ctx,
    );
    assert.equal(gamma.details.ok, true, `gamma was not born: ${JSON.stringify(gamma.details)}`);
    const gammaAfterBirth = stateOf(sandbox);
    const gammaNode = gammaAfterBirth.nodes.find((nd) => nd.id === gamma.details.id);
    assert.ok(gammaNode, `gamma is absent from state: ${JSON.stringify(gamma.details)}`);
    assert.equal(gammaNode.leaves.length, 3, `gamma's source leaves were not all grouped: ${JSON.stringify({ result: gamma.details, node: gammaNode })}`);
    const beforeLaterParent = stateOf(sandbox);
    const heldBeforeLaterParent = new Set(beforeLaterParent.oldNode.nodes ?? []);
    let closureGrew = true;
    while (closureGrew) {
      closureGrew = false;
      for (const nd of beforeLaterParent.nodes) {
        if (!heldBeforeLaterParent.has(nd.id)) continue;
        for (const child of nd.children ?? []) {
          if (heldBeforeLaterParent.has(child)) continue;
          heldBeforeLaterParent.add(child);
          closureGrew = true;
        }
      }
    }
    const lastHeldIndex = Math.max(...beforeLaterParent.nodes
      .map((nd, index) => heldBeforeLaterParent.has(nd.id) ? index : -1));
    const gammaIndex = beforeLaterParent.nodes.findIndex((nd) => nd.id === gamma.details.id);
    assert.ok(gammaIndex > lastHeldIndex, 'the later child is not after the existing pit closure');
    const laterParent = await tools.get('cwl_group').execute(
      't', { pit: true, nodes: [gamma.details.id], name: 'later', description: 'LATER follows the existing archive' }, undefined, undefined, ctx,
    );
    assert.equal(laterParent.details.ok, true, `the later pit parent was refused: ${JSON.stringify(laterParent.details)}`);
    const afterLaterParent = stateOf(sandbox);
    assert.equal(afterLaterParent.nodes.findIndex((nd) => nd.id === laterParent.details.id), lastHeldIndex + 1,
      'the new pit root was inserted before an existing contained descendant');

    // The membership checks must also recognize descendants: description rewrites and pit
    // cataloguing work on nested nodes/leaves, and a later merge never absorbs them again.
    const rewritten = await tools.get('cwl_group').execute(
      't', { pit: true, node: alpha.details.id, description: 'ALPHA updated inside the pit' }, undefined, undefined, ctx,
    );
    assert.equal(rewritten.details.ok, true, `rewriting a nested pit topic failed: ${JSON.stringify(rewritten.details)}`);
    assert.equal(rewritten.details.inPit, true);
    const nestedLeaves = [...(parent.children ?? [])]
      .flatMap((id) => st.nodes.find((nd) => nd.id === id)?.leaves ?? [])
      .slice(0, 3);
    const catalogue = await tools.get('cwl_group').execute(
      't', { pit: true, leaves: nestedLeaves, name: 'nested-catalogue', description: 'Nested leaves catalogued' }, undefined, undefined, ctx,
    );
    assert.equal(catalogue.details.ok, true, `cataloguing nested pit leaves failed: ${JSON.stringify(catalogue.details)}`);
    const nextMerge = await tools.get('cwl_old').execute('t', { text: 'another merge' }, undefined, undefined, ctx);
    const mergedNodes = Array.isArray(nextMerge.details.nodes) ? nextMerge.details.nodes : [];
    assert.ok(!mergedNodes.some((id) => [alpha.details.id, beta.details.id].includes(id)),
      'a nested pit node was treated as young and absorbed a second time');
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

test('rewriting a pit topic description does not leak the i18n source', async () => {
  // Found by USING the tool, not by reading it: the reply concatenated `t('groupDescriptionUpdated')`
  // instead of CALLING it, so the agent got the source of the arrow function — `() => ...` and the
  // whole Italian sentence — where the sentence itself belonged. `groupDescriptionUpdated` is a
  // FUNCTION (`() => string`), and every other i18n entry in a reply is called.
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await makeLeaves(sandbox, tools, hooks, ctx, 14);
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the first stories' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const inPit = stateOf(sandbox).nodes.filter((nd) => (stateOf(sandbox).oldNode.nodes ?? []).includes(nd.id)).flatMap((nd) => nd.leaves);
    const topic = await tools.get('cwl_group').execute(
      't', { pit: true, leaves: inPit.slice(0, 3), name: 'catalogue', description: 'CATALOGUE holds three stories' }, undefined, undefined, ctx,
    );
    assert.equal(topic.details.ok, true, `the pit topic was refused: ${JSON.stringify(topic.details)}`);
    const rewritten = await tools.get('cwl_group').execute(
      't', { pit: true, node: topic.details.id, description: 'CATALOGUE now covers three stories' }, undefined, undefined, ctx,
    );
    assert.equal(rewritten.details.ok, true, `the rewrite was refused: ${JSON.stringify(rewritten.details)}`);
    assert.equal(rewritten.details.descriptionUpdated, true, 'the rewrite was not recorded');
    const text = String(rewritten.content[0].text);
    assert.doesNotMatch(text, /=>/, `the reply leaked the source of the i18n entry: ${text.slice(0, 200)}`);
    assert.match(text, /NOTHING in the index|NON cambia niente/, 'the sentence itself is missing from the reply');
  } finally {
    home.restore(); sandbox.cleanup();
  }
});
