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
 *  2. a leaf that is not in the buffer is accepted -> material older than the frontier would
 *     be rearranged under the agent's hands;
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

/** `young 2n/9l` from the index line: the nodes and the labels the head is made of. */
const youngOf = (line) => {
  const m = String(line).match(/young (\d+)n\/(\d+)l/);
  return m ? { nodes: Number(m[1]), leaves: Number(m[2]) } : null;
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
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 6);
    const inBuffer = st.spans.slice(0, 5).map((s) => s.id);
    const res = await group(tools, ctx, { leaves: inBuffer, name: 'too-tiny', description: 'A topic that cannot pay for itself.' });
    assert.equal(res.details.ok, false, 'a topic below the guard was born');
    assert.equal(res.details.error, 'too-small');
    const body = text(res);
    assert.ok(body.includes('10800'), `the refusal does not say what it needed: ${body}`);
    assert.ok(body.includes(String(res.details.microChars)), `the refusal does not say what it would free: ${body}`);

    const after = youngOf(await statusText(tools, ctx));
    assert.equal(after.nodes, 1, 'a refused birth left a node behind');
    assert.equal(st.spans.length, 6, 'a refused birth touched the leaves');
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

test('only the leaves of the buffer can be moved', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ nodeCapacity: 2, mergeMinRatio: 0, mergeMinChars: 0 });
  try {
    const st = await leavesWithMicros(sandbox, hooks, ctx, tools, 4);
    // nodeCapacity 2 and looseLeaves 1: nodes are [leaf0, leaf1] and the buffer [leaf2];
    // leaf3 is loose. So leaf2 is the one legal to move, and leaf0 is not.
    const born = await group(tools, ctx, { leaves: [st.spans[2].id], name: 'frontier', description: 'Born from the frontier.' });
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const res = await group(tools, ctx, { node: born.details.id, leaves: [st.spans[0].id] });
    assert.equal(res.details.ok, false, 'a leaf older than the frontier was moved');
    assert.equal(res.details.why, 'leaf-not-in-the-buffer', `unexpected refusal: ${JSON.stringify(res.details)}`);
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
    assert.match(line, /topics 1/, `the topic is not counted in the index line: ${line}`);
    assert.match(line, /young 2n/, 'the buffer behind the topic disappeared from the line');
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
    const youngShape = (line) => ((line.match(/young (\d+)n\/(\d+)l/) ?? ['', '?', '?']).slice(1, 3).join('/'));
    assert.equal(youngShape(after), youngShape(before), `the shape of the index moved across the restart: ${youngShape(before)} -> ${youngShape(after)}`);
    const page = await openPage(run2.tools, ctx, born.details.id);
    assert.ok(page.includes(DESCRIPTION), `the description did not survive the restart: ${page.slice(0, 200)}`);
  } finally {
    home.restore();
  }
});
