/**
 * TOPIC NODES — the nested archive:
 *
 *   [OLD NODE (the pit)] [topic nodes and legacy nodes] [BUFFER (the last node)] [open leaves]
 *
 * A topic node is born COLLAPSED: `name` and `description` are written once, at birth, and
 * the description stands for its leaves in the index from that moment on. The description
 * must already cover the FUTURE use of the topic, and that is not a style rule: it is what
 * makes adding leaves later free — their labels leave the head, the description stays, and
 * no synthesis is written a second time. Rewriting it would move the prefix of the index.
 *
 * WHAT THE TESTS CAN SEE. The nodes ARE persisted: `saveState` writes the graph, the spans,
 * the graveyard, the pit and the node structure (id, leaves, name, description), and
 * `refreshNodes` keeps the nodes it finds, pruning only the leaves that lost their micro.
 * Persisting them is not an optimization: a topic's name cannot be recomputed from a leaf,
 * and without the pit's node ids the old-node page would come back EMPTY while the leaves it
 * had absorbed were re-formed as young ones, back in the head under a description that no
 * longer exists. So the shape is still observed through the tools — `cwl_status` (the index
 * line), `cwl_group` (its details) and `cwl_open` (the page of a node) — and one test kills
 * the in-memory state to prove that the file, and only the file, carries the catalogue.
 *
 * THE FIVE WAYS THIS TEST MUST DIE:
 *  1. the topic replaces the buffer instead of sitting BEHIND it -> the first node would be
 *     a topic, which could not act as a buffer;
 *  2. a SCATTERED set of leaves is accepted -> the topic would not be one block in a
 *     chronological index, and the leaves left between its own would have nowhere to go;
 *  3. a leaf already inside the old node is accepted -> the pit's synthesis stands for it,
 *     and the same content would be described twice;
 *  4. a topic below the size guard is born -> the description costs more than it frees;
 *  5. the buffer dies when its last leaf moves into a topic -> the first node would become
 *     a topic node, which the design forbids.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const DESCRIPTION = 'Everything about the OTP login: the flow, the choices made, and the traps.';

const config = (extra = {}) => ({
  // A LOW budget on purpose: `cwl_compress_range` only offers a range when the context is
  // over the threshold, so a generous budget answers `nothing-to-compress` and the fixture
  // never builds a leaf. The leaves are `custom` and the eviction protects them.
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  looseLeaves: 1,
  nodeCapacity: 30,
  mergeNodesAt: 3,
  ...extra,
});

async function boot(extra) {
  const sandbox = makeSandbox({ name: `topic-${seq++}`, config: config(extra) });
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

const text = (res) => res.content.map((c) => c.text).join('\n');

const statusText = async (tools, ctx) =>
  text(await tools.get('cwl_status').execute('t', {}, undefined, undefined, ctx));

const openPage = async (tools, ctx, id) =>
  text(await tools.get('cwl_open').execute('t', { id }, undefined, undefined, ctx));

/** The young part of the index line — `topics 2n/9l │ buffer 1n/0l │ ordinary 0n/0l` — summed:
 *  the nodes and the labels the head is made of outside the pit. The line keeps the three
 *  apart because the head pays for them differently (a topic costs ONE description, the buffer
 *  is the working area, an ordinary node costs the labels of its leaves); what the tests below
 *  watch is their SUM, so a grouping shows up as leaves moving from one group to another. */
const youngOf = (line) => {
  const m = String(line).match(
    /topics (\d+)n\/(\d+)l │ buffer (\d+)n\/(\d+)l │ ordinary (\d+)n\/(\d+)l/,
  );
  return m
    ? { nodes: Number(m[1]) + Number(m[3]) + Number(m[5]), leaves: Number(m[2]) + Number(m[4]) + Number(m[6]) }
    : null;
};

const group = (tools, ctx, args) => tools.get('cwl_group').execute('t', args, undefined, undefined, ctx);

/** Builds `n` leaves, gives each a micro of ~1,300 characters, and forms the nodes. */
async function leavesWithMicros(sandbox, hooks, ctx, tools, n) {
  for (let i = 1; i <= n; i++) {
    await hook(hooks, ctx, conversation(1, i + 3));
    const res = await tools.get('cwl_compress_range').execute(
      't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
    );
    assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
  }
  const leaves = stateOf(sandbox).spans;
  assert.equal(leaves.length, n, `expected ${n} leaves, ${leaves.length} in the state`);
  for (let i = 0; i < n; i++) {
    const r = await tools.get('cwl_micro').execute(
      't', { id: leaves[i].id, text: `MICRO-${i + 1} ` + 'y'.repeat(1300) }, undefined, undefined, ctx,
    );
    assert.equal(r.details.ok, true, `micro on leaf ${i + 1}: ${JSON.stringify(r.details)}`);
  }
  await hook(hooks, ctx, conversation(1, 40));
  return stateOf(sandbox);
}

test('a topic is born collapsed, behind the buffer', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    // looseLeaves 1: the tenth leaf stays loose, the other nine sit in the buffer.
    const inBuffer = st.spans.slice(0, 9).map((s) => s.id);
    const before = youngOf(await statusText(tools, ctx));
    assert.ok(before && before.nodes === 1, `expected one node before the topic, got ${JSON.stringify(before)}`);

    const res = await group(tools, ctx, { leaves: inBuffer, name: 'login-otp', description: DESCRIPTION });
    assert.equal(res.details.ok, true, `the topic was not born: ${JSON.stringify(res.details)}`);
    assert.equal(res.details.name, 'login-otp');
    assert.equal(res.details.leaves, 9);

    const after = youngOf(await statusText(tools, ctx));
    assert.ok(after, 'the index line lost its shape');
    assert.equal(after.nodes, 2, 'the topic replaced the buffer: the buffer must survive BEHIND it');
    assert.equal(after.leaves, 9, `the head should hold the 9 labels of the topic, it holds ${after.leaves}`);

    const page = await openPage(tools, ctx, res.details.id);
    assert.ok(page.includes('login-otp'), `the topic page lost the name: ${page.slice(0, 200)}`);
    assert.ok(page.includes(DESCRIPTION), `the topic page lost the description: ${page.slice(0, 200)}`);
    assert.equal((page.match(/^- sp-/gm) || []).length, 9, 'the topic page does not list its nine leaves');
  } finally {
    home.restore();
  }
});

test('a topic below the size guard is refused, and the numbers are said', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // FOUR leaves in the buffer, not five: with `mergeMinRatio` at 1.8 the floor is 6,480
    // characters, and five micros of ~1,309 hold ~6,545 — only 65 over, so the topic would
    // be BORN and this test would assert a refusal that never happens. Four hold ~5,236,
    // comfortably below. The margin is a coincidence of the fixture, not a property of the
    // code: if the floor moves again, re-measure instead of guessing.
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 5);
    const inBuffer = st.spans.slice(0, 4).map((s) => s.id);
    const res = await group(tools, ctx, { leaves: inBuffer, name: 'too-tiny', description: 'A topic that cannot pay for itself.' });
    assert.equal(res.details.ok, false, 'a topic below the guard was born');
    assert.equal(res.details.error, 'too-small');
    const body = text(res);
    assert.ok(body.includes('6480'), `the refusal does not say what it needed: ${body}`);
    assert.ok(body.includes(String(res.details.microChars)), `the refusal does not say what it would free: ${body}`);

    const after = youngOf(await statusText(tools, ctx));
    assert.equal(after.nodes, 1, 'a refused birth left a node behind');
    assert.equal(st.spans.length, 5, 'a refused birth touched the leaves');
  } finally {
    home.restore();
  }
});

test('the description is written once: adding leaves does not change it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    const inBuffer = st.spans.slice(0, 9).map((s) => s.id);
    const born = await group(tools, ctx, { leaves: inBuffer, name: 'login-otp', description: DESCRIPTION });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const loose = st.spans[9].id;
    const added = await group(tools, ctx, { node: born.details.id, leaves: [loose] });
    assert.equal(added.details.ok, true, `adding a leaf was refused: ${JSON.stringify(added.details)}`);

    const page = await openPage(tools, ctx, born.details.id);
    assert.ok(page.includes(DESCRIPTION), `the description was rewritten: ${page.slice(0, 240)}`);
    assert.equal((page.match(/^- sp-/gm) || []).length, 10, 'the added leaf is not in the topic');
  } finally {
    home.restore();
  }
});

test('an ordinary node is a source too, and what it keeps is re-partitioned in time', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 2, looseLeaves: 0, mergeMinRatio: 0, mergeMinChars: 0 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 5);
    // nodeCapacity 2 and looseLeaves 0: [l0,l1] [l2,l3] [l4] — and only the LAST one is the
    // buffer. The first node is ordinary, and its leaves are still in the head as micros.
    const before = stateOf(sandbox).nodes;
    assert.deepEqual(before.map((nd) => nd.leaves.length), [2, 2, 1], `unexpected shape: ${JSON.stringify(before.map((nd) => nd.leaves))}`);
    // A leaf of the FIRST node, which is neither the buffer nor loose: under the old rule this
    // was refused outright, and that is exactly what `groupBeyondBuffer` opens up.
    const born = await group(tools, ctx, { leaves: [st.spans[0].id], name: 'back', description: 'Older material, taken from a closed node.' });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const after = stateOf(sandbox).nodes;
    // The topic lands at the position of the node it took the leaf from, and the leaf left
    // behind is in a node AFTER it: keeping the chronology is the whole point of the re-partition.
    const topicIdx = after.findIndex((nd) => nd.id === born.details.id);
    assert.equal(topicIdx, 0, `the topic was not born at the first node's place: ${JSON.stringify(after.map((nd) => nd.id))}`);
    assert.deepEqual(after[0].leaves, [st.spans[0].id], 'the topic does not hold the leaf it took');
    const restIdx = after.findIndex((nd) => nd.leaves.includes(st.spans[1].id));
    assert.ok(restIdx > topicIdx, `the leaf left behind moved in time: node ${restIdx}, topic ${topicIdx}`);
    // The buffer is still the LAST node and still holds its own leaf.
    assert.deepEqual(after[after.length - 1].leaves, [st.spans[4].id], 'the buffer is no longer last');
  } finally {
    home.restore();
  }
});

test('with groupBeyondBuffer off, only the leaves of the buffer can be moved', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 2, looseLeaves: 0, mergeMinRatio: 0, mergeMinChars: 0, groupBeyondBuffer: false });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 5);
    const res = await group(tools, ctx, { leaves: [st.spans[0].id], name: 'back', description: 'Refused under the old rule.' });
    assert.equal(res.details.ok, false, 'a leaf older than the frontier was moved with the key off');
    assert.equal(res.details.why, 'leaf-not-in-the-buffer', `unexpected refusal: ${JSON.stringify(res.details)}`);
  } finally {
    home.restore();
  }
});

test('a topic is never a source: its description stands for its leaves', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 2, looseLeaves: 0, mergeMinRatio: 0, mergeMinChars: 0 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 5);
    const born = await group(tools, ctx, { leaves: [st.spans[0].id, st.spans[1].id], name: 'first', description: 'A topic over the first node.' });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const res = await group(tools, ctx, { node: born.details.id, leaves: [st.spans[0].id] });
    assert.equal(res.details.ok, false, 'a leaf was taken back out of a topic');
    assert.equal(res.details.why, 'leaf-in-a-topic', `unexpected refusal: ${JSON.stringify(res.details)}`);
  } finally {
    home.restore();
  }
});

test('the topic is one block: scattered leaves are refused', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 2, looseLeaves: 0, mergeMinRatio: 0, mergeMinChars: 0 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 5);
    // l0 and l2 are in different nodes and l1 sits between them: a topic holding both would
    // not be one block, and l1 would have nowhere to go that is both inside its span and out.
    const res = await group(tools, ctx, { leaves: [st.spans[0].id, st.spans[2].id], name: 'scattered', description: 'Two leaves with one in between.' });
    assert.equal(res.details.ok, false, 'a scattered set of leaves was accepted');
    assert.equal(res.details.why, 'leaves-not-contiguous', `unexpected refusal: ${JSON.stringify(res.details)}`);
  } finally {
    home.restore();
  }
});

test('a leaf already inside the old node cannot be grouped', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({
    nodeCapacity: 2, mergeNodesAt: 2, mergeMinRatio: 0, mergeMinChars: 0,
  });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 4);
    const merged = await tools.get('cwl_old').execute('t', { text: 'MERGE-SUMMARY: the oldest material.' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    assert.equal(merged.details.leaves, 2, `the merge took ${merged.details.leaves} leaves, expected the oldest node's 2`);
    // The pit absorbed the oldest node, which is [leaf0, leaf1].
    const res = await group(tools, ctx, { leaves: [st.spans[0].id], name: 'from-the-pit', description: 'This leaf left the frontier long ago.' });
    assert.equal(res.details.ok, false, 'a leaf of the pit was grouped: the pit stands for it');
    assert.equal(res.details.why, 'leaf-in-the-pit', `unexpected refusal: ${JSON.stringify(res.details)}`);
  } finally {
    home.restore();
  }
});

test('the buffer survives when its last leaf moves into a topic', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ mergeMinRatio: 0, mergeMinChars: 0 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 3);
    // The third leaf is loose; leaves 0 and 1 are the buffer, and both move into the topic.
    const born = await group(tools, ctx, { leaves: [st.spans[0].id, st.spans[1].id], name: 'everything', description: 'All the frontier material, archived.' });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    // A turn later `refreshNodes` runs, and it prunes empty nodes: the buffer must survive,
    // or the first node would become a topic node.
    await hook(hooks, ctx, conversation(1, 40));
    const after = youngOf(await statusText(tools, ctx));
    assert.ok(after, 'the index line lost its shape');
    assert.equal(after.nodes, 2, 'the buffer was pruned: the first node is now a topic node');
  } finally {
    home.restore();
  }
});

test('the description replaces the labels of the topic in the head', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    const inBuffer = st.spans.slice(0, 9).map((s) => s.id);
    const hasLabel = (list, n) => list.some((m) => String(m.content ?? '').includes(`MICRO-${n} `));

    // Before the topic: NINE labels are in the head, and the context pays for all nine.
    let messages = await hook(hooks, ctx, conversation(1, 40));
    assert.ok(hasLabel(messages, 1) && hasLabel(messages, 9), 'the fixture did not put the nine labels in the head');

    const born = await group(tools, ctx, { leaves: inBuffer, name: 'login-otp', description: DESCRIPTION });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    messages = await hook(hooks, ctx, conversation(1, 40));

    // After: ONE description, and none of the nine labels. This is the saving of the whole
    // nesting — without it, grouping nine leaves would still show nine labels.
    const shown = messages.filter((m) => String(m.content ?? '').includes(DESCRIPTION));
    assert.equal(shown.length, 1, `the description is injected ${shown.length} time(s), expected exactly once`);
    const left = [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((n) => hasLabel(messages, n));
    assert.equal(left.length, 0, `the labels of the topic are still in the head: MICRO-${left.join(', MICRO-')}`);
    // The tenth leaf stayed loose, so its own label is still there: it is not the topic's.
    assert.ok(hasLabel(messages, 10), 'the loose leaf lost its label');
  } finally {
    home.restore();
  }
});

/** One more leaf, built after the rest: the frontier keeps producing material. */
async function oneMoreLeaf(sandbox, hooks, ctx, tools, n) {
  const before = stateOf(sandbox).spans.length;
  await hook(hooks, ctx, conversation(1, n + 3));
  const res = await tools.get('cwl_compress_range').execute(
    't', { summary: `EXTRA-${n} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
  );
  assert.equal(res.details.ok, true, `the extra leaf was not born: ${JSON.stringify(res.details)}`);
  const spans = stateOf(sandbox).spans;
  assert.equal(spans.length, before + 1, 'the extra leaf did not arrive');
  const id = spans[spans.length - 1].id;
  const r = await tools.get('cwl_micro').execute(
    't', { id, text: `MICRO-EXTRA-${n} ` + 'y'.repeat(1300) }, undefined, undefined, ctx,
  );
  assert.equal(r.details.ok, true, `micro on the extra leaf: ${JSON.stringify(r.details)}`);
  await hook(hooks, ctx, conversation(1, n + 10));
  return id;
}

test('a topic inside the old node keeps receiving leaves, and only there is its description rewritten', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({
    nodeCapacity: 2, mergeNodesAt: 2, mergeMinRatio: 0, mergeMinChars: 0,
  });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 4);
    const born = await group(tools, ctx, { leaves: [st.spans[2].id], name: 'frontier', description: 'FIRST-DESCRIPTION: born at the frontier.' });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const merged = await tools.get('cwl_old').execute('t', { text: 'MERGE-SUMMARY: the legacy material.' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    assert.equal(merged.details.topicsConcatenated, 1, 'the immutable description was not glued to the pit synthesis');

    // The topic is inside the pit now, and a leaf from the frontier may still join it.
    const extra = await oneMoreLeaf(sandbox, hooks, ctx, tools, 40);
    const added = await group(tools, ctx, {
      node: born.details.id, leaves: [extra], description: 'SECOND-DESCRIPTION: the frontier work joined this topic.',
    });
    assert.equal(added.details.ok, true, `adding to a topic inside the pit was refused: ${JSON.stringify(added.details)}`);
    assert.equal(added.details.inPit, true, 'the tool does not know the topic is in the pit');
    assert.equal(added.details.descriptionUpdated, true, 'the description was not rewritten');

    // And the context does NOT move: what stands for the topic in the head is the pit's
    // synthesis, frozen when it was written. The living copy is on the topic's own page.
    const messages = await hook(hooks, ctx, conversation(1, 60));
    const joined = messages.map((m) => String(m.content ?? '')).join('\n');
    assert.ok(joined.includes('FIRST-DESCRIPTION'), 'the pit synthesis lost what it was written with');
    assert.ok(!joined.includes('SECOND-DESCRIPTION'), 'rewriting the description MOVED the context');
    const page = await openPage(tools, ctx, born.details.id);
    assert.ok(page.includes('SECOND-DESCRIPTION'), `the living copy is not on the topic page: ${page.slice(0, 240)}`);
  } finally {
    home.restore();
  }
});

test('a description outside the old node cannot be rewritten', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    const born = await group(tools, ctx, { leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const res = await group(tools, ctx, { node: born.details.id, description: 'A rewritten description.' });
    assert.equal(res.details.ok, false, 'the description of a topic OUTSIDE the pit was rewritten');
    assert.equal(res.details.why, 'description-is-immutable', `unexpected refusal: ${JSON.stringify(res.details)}`);
    const page = await openPage(tools, ctx, born.details.id);
    assert.ok(page.includes(DESCRIPTION), 'the refused rewrite still changed the description');
  } finally {
    home.restore();
  }
});

test('the index line counts the topics', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    assert.match(await statusText(tools, ctx), /topics 0/, 'a fresh index should count no topics');
    const born = await group(tools, ctx, { leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const line = await statusText(tools, ctx);
    // The three groups, checked APART: a topic holds its labels and injects ONE description, the
    // buffer survives behind it, and nothing is left in the ordinary group. Asserting only a
    // total would not say which of the three moved.
    assert.match(line, /topics 1n\/9l/, `the topic is not counted with the labels it holds: ${line}`);
    assert.match(line, /buffer 1n\/0l/, `the buffer behind the topic disappeared from the line: ${line}`);
    assert.match(line, /ordinary 0n\/0l/, `the labels of the topic are still counted as ordinary: ${line}`);
  } finally {
    home.restore();
  }
});

test('the index line keeps all its sections, in order', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    const line = await statusText(tools, ctx);
    // `indexShape` hands its numbers to `t('indexLine')` POSITIONALLY, and a positional tuple is
    // not protected by the typecheck: nothing notices a section added, dropped or reordered
    // except a test that reads the line as a whole. This is that test. It is here because the
    // operator reads this line every round, and a section silently disappearing would take a
    // measurement away without a single failure.
    assert.match(
      line,
      /pit \d+n\/\d+l │ topics \d+n\/\d+l │ buffer \d+n\/\d+l │ ordinary \d+n\/\d+l │ loose \d+ │ waiting micro \d+ │ head ~[\d,.]+t │ \d+ evict │ [\d,.]+ saved/,
      `the index line lost its shape: ${line}`,
    );
  } finally {
    home.restore();
  }
});

test('the status names the topics, not just counts them', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    assert.doesNotMatch(await statusText(tools, ctx), /Topics \(/, 'a fresh index should list no topic');
    const born = await group(tools, ctx, { leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const line = await statusText(tools, ctx);
    assert.match(line, /Topics \(1\): login-otp/, `the status does not name the topic: ${line}`);
  } finally {
    home.restore();
  }
});

test('a topic survives a restart: the file carries the catalogue', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 10);
    const born = await group(tools, ctx, {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const before = await statusText(tools, ctx);
    assert.match(before, /Topics \(1\): login-otp/, `before the restart: ${before}`);

    // A NEW module over the SAME home and the same session: all that survives is the file.
    // Without `nodes` in the payload the topic is gone here, and its nine leaves are
    // re-formed as young ones: back in the head, with no description standing for them.
    const run2 = await bootExtension(sandbox, { name: 'run2' });
    await run2.hooks.get('session_start')({}, ctx);
    const after = await statusText(run2.tools, ctx);
    assert.match(after, /Topics \(1\): login-otp/, `the topic lost its name after the restart: ${after}`);
    // The SHAPE must not move across the restart: a topic is a young node too (it is outside
    // the pit), so the count has to match, not vanish. Compared as a STRING, because
    // `youngOf` hands back a parsed shape, not a primitive. What proves the fix is the name.
    const youngShape = (line) => (String(line).match(/topics (\d+)n\/(\d+)l │ buffer (\d+)n\/(\d+)l │ ordinary (\d+)n\/(\d+)l/) ?? []).slice(1).join('/');
    assert.equal(youngShape(after), youngShape(before), `the shape of the index moved across the restart: ${youngShape(before)} -> ${youngShape(after)}`);
    const page = await openPage(run2.tools, ctx, born.details.id);
    assert.ok(page.includes(DESCRIPTION), `the description did not survive the restart: ${page.slice(0, 200)}`);
  } finally {
    home.restore();
  }
});

test('a topic can be born INSIDE the old node: 3 leaves are enough, and the pit does not move', async () => {
  // A small capacity and a cheap merge, so that a pit exists without a huge fixture. The
  // frontier guard is relaxed on purpose: the pit path must not be sneaking through it.
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 3, mergeMinRatio: 1, mergeMinChars: 100 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 12);
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS-BEFORE-THE-CATALOGUE' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const pit = merged.details.id;
    const before = await openPage(tools, ctx, pit);
    assert.ok(before.includes('SYNTHESIS-BEFORE-THE-CATALOGUE'), `the merge summary is not on the page: ${before.slice(0, 200)}`);

    // The oldest leaves are the ones the pit absorbed: the nodes are formed in time order.
    const inside = st.spans.slice(0, 3).map((s) => s.id);
    const thin = await group(tools, ctx, { leaves: inside.slice(0, 2), name: 'too-thin', description: DESCRIPTION, pit: true });
    assert.equal(thin.details.why, 'pit-too-few', `two leaves were accepted inside the pit: ${JSON.stringify(thin.details)}`);

    const born = await group(tools, ctx, { leaves: inside, name: 'pit-catalogue', description: DESCRIPTION, pit: true });
    assert.equal(born.details.ok, true, `the pit topic was not born: ${JSON.stringify(born.details)}`);
    assert.equal(born.details.pit, true, 'the response does not say the topic went into the pit');

    const after = await openPage(tools, ctx, pit);
    assert.ok(after.includes('SYNTHESIS-BEFORE-THE-CATALOGUE'), `the pit synthesis was rewritten: ${after.slice(0, 300)}`);
    assert.match(after, /TOPIC "pit-catalogue"/, `the pit page does not name the new topic: ${after.slice(0, 400)}`);
    assert.match(await statusText(tools, ctx), /Topics \(1\): pit-catalogue/, 'the catalogue does not list it');
  } finally {
    home.restore();
  }
});

test('the pit can be re-catalogued after the fact: a fat node is split into topics, no leaf is left behind', async () => {
  // `nodeCapacity: 6` so that a pit node is big enough to be split into two topics of three,
  // and a cheap merge so the pit exists without a huge fixture.
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 6, mergeMinRatio: 1, mergeMinChars: 100 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 18);
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS-THAT-MUST-NOT-MOVE' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const pit = merged.details.id;

    const shapeOf = (line) => (line.match(/pit (\d+)n\/(\d+)l/) ?? []).slice(1, 3).join('/');
    const before = await statusText(tools, ctx);
    const pageBefore = await openPage(tools, ctx, pit);
    const nodesBefore = (pageBefore.match(/^- nd-[0-9a-f]+/gm) ?? []).length;
    assert.ok(nodesBefore >= 1, `the pit page lists no node: ${pageBefore.slice(0, 300)}`);

    // The oldest leaves are the ones the pit absorbed, in the order the nodes were formed.
    const fat = st.spans.slice(0, 6).map((s) => s.id);
    const first = await group(tools, ctx, { leaves: fat.slice(0, 3), name: 'first-half', description: DESCRIPTION, pit: true });
    const second = await group(tools, ctx, { leaves: fat.slice(3, 6), name: 'second-half', description: DESCRIPTION, pit: true });
    assert.equal(first.details.ok, true, `the first half was not born: ${JSON.stringify(first.details)}`);
    assert.equal(second.details.ok, true, `the second half was not born: ${JSON.stringify(second.details)}`);

    const after = await statusText(tools, ctx);
    const pageAfter = await openPage(tools, ctx, pit);

    // THE INVARIANT THE OPERATOR ASKED FOR: every leaf that was in the pit is still in the pit.
    // Only the number of NODES changes — the fat one is gone, two topics take its place.
    assert.equal(shapeOf(after).split('/')[1], shapeOf(before).split('/')[1],
      `the leaves in the pit changed: ${shapeOf(before)} -> ${shapeOf(after)}`);
    assert.match(pageAfter, /TOPIC "first-half"/, `the pit page lost the first topic: ${pageAfter.slice(0, 400)}`);
    assert.match(pageAfter, /TOPIC "second-half"/, `the pit page lost the second topic: ${pageAfter.slice(0, 400)}`);
    assert.ok(pageAfter.includes('SYNTHESIS-THAT-MUST-NOT-MOVE'), `the pit synthesis was rewritten: ${pageAfter.slice(0, 300)}`);
    assert.match(after, /Topics \(2\): /, `the catalogue does not list the two topics: ${after}`);

    // The page's header counts the nodes the pit CLAIMS to hold; the index line counts the ones
    // that really exist. An emptied node must leave BOTH, or the page starts lying.
    const claimed = Number((pageAfter.match(/— (\d+) node\(s\) inside/) ?? [])[1]);
    assert.equal(claimed, Number(shapeOf(after).split('/')[0]),
      `the pit page claims ${claimed} node(s) but the index counts ${shapeOf(after).split('/')[0]}`);
  } finally {
    home.restore();
  }
});
