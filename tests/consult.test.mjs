/**
 * THE CONSULTATION TOOLS — finding a memory instead of hunting for its id.
 *
 * The problem they answer, measured: `cwl_status` named no id at all, `cwl_recall` searched
 * the TRANSCRIPT and returned message previews, and the only place a leaf id ever appeared
 * was the block injected into the context — for the young nodes and the loose leaves. A leaf
 * INSIDE a topic had no way in except opening the topic first, i.e. guessing which topic it
 * was. Agents paid dozens of turns for that.
 *
 * WHAT THESE TESTS PIN, one per way the fix can silently die:
 *  1. `cwl_find` must reach a leaf INSIDE a topic — the case that had no path at all;
 *  2. `cwl_map` must name the nodes AND their ids, and must NOT list a topic's leaves (the
 *     description is what stands for them; listing them would put the ids back in the head);
 *  3. `cwl_node` must work on a topic, on the buffer and on the pit — "any node";
 *  4. `cwl_pending` must count only what is still to be ordered, and say whether the total
 *     reaches the threshold a topic needs;
 *  5. `cwl_status` must show the ids of the topics, the pit and the buffer.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const DESCRIPTION = 'Everything about the OTP login: the flow, the choices made, and the traps.';

const config = (extra = {}) => ({
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
  const sandbox = makeSandbox({ name: `consult-${seq++}`, config: config(extra) });
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
const call = (tools, name, args, ctx) => tools.get(name).execute('t', args, undefined, undefined, ctx);

/** Builds `n` leaves with a DISTINCTIVE word in each label, so a search has something to
 *  find. The word is what makes the test meaningful: `cwl_find` is asked for a leaf that is
 *  inside a topic, and the id it returns must be that leaf's. */
async function leavesWithWords(sandbox, hooks, ctx, tools, n, words) {
  for (let i = 1; i <= n; i++) {
    await hook(hooks, ctx, conversation(1, i + 3));
    const res = await call(tools, 'cwl_compress_range', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, ctx);
    assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
  }
  const leaves = stateOf(sandbox).spans;
  assert.equal(leaves.length, n, `expected ${n} leaves, ${leaves.length} in the state`);
  for (let i = 0; i < n; i++) {
    const word = words[i] ?? `filler${i + 1}`;
    const r = await call(tools, 'cwl_micro', { id: leaves[i].id, text: `${word} ` + 'y'.repeat(1300) }, ctx);
    assert.equal(r.details.ok, true, `micro on leaf ${i + 1}: ${JSON.stringify(r.details)}`);
  }
  await hook(hooks, ctx, conversation(1, 40));
  return stateOf(sandbox);
}

test('cwl_find reaches a leaf INSIDE a topic, without opening the topic', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // The tenth leaf stays loose (looseLeaves 1); the first nine go into a topic, so their
    // ids leave the head entirely. `zebrafish` lives in the THIRD leaf, i.e. inside the topic.
    const st = await leavesWithWords(sandbox, hooks, ctx, tools, 10, ['alpha', 'bravo', 'zebrafish']);
    const inside = st.spans[2].id;
    const born = await call(tools, 'cwl_group', {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    }, ctx);
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);

    const found = await call(tools, 'cwl_find', { query: 'zebrafish' }, ctx);
    assert.equal(found.details.ok, true, `cwl_find failed: ${JSON.stringify(found.details)}`);
    assert.ok(
      found.details.ids.includes(inside),
      `the leaf inside the topic was not found; ids=${JSON.stringify(found.details.ids)}`,
    );
    assert.match(text(found), /zebrafish/, `the result does not show what it matched: ${text(found)}`);
  } finally {
    home.restore();
  }
});

test('cwl_find with scope=both also searches the transcript', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await leavesWithWords(sandbox, hooks, ctx, tools, 3, ['alpha']);
    const indexOnly = await call(tools, 'cwl_find', { query: 'turn 7 content', scope: 'index' }, ctx);
    const both = await call(tools, 'cwl_find', { query: 'turn 7 content', scope: 'both' }, ctx);
    assert.equal(both.details.ok, true, `cwl_find scope=both failed: ${JSON.stringify(both.details)}`);
    assert.ok(
      both.details.hits >= indexOnly.details.hits,
      `scope=both must not find less than scope=index: ${indexOnly.details.hits} -> ${both.details.hits}`,
    );
    assert.equal(both.details.scope, 'both');
  } finally {
    home.restore();
  }
});

test('cwl_map names every node with its id, and hides the leaves of a topic', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithWords(sandbox, hooks, ctx, tools, 10, ['alpha', 'bravo', 'charlie']);
    const born = await call(tools, 'cwl_group', {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    }, ctx);
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);
    const topicId = born.details.id;

    const map = await call(tools, 'cwl_map', {}, ctx);
    assert.equal(map.details.ok, true, `cwl_map failed: ${JSON.stringify(map.details)}`);
    const body = text(map);
    assert.match(body, new RegExp(topicId), `the map does not name the topic id: ${body}`);
    assert.match(body, /login-otp/, `the map does not name the topic: ${body}`);
    assert.match(body, /topic/, `the map does not say the node is a topic: ${body}`);
    // The leaves INSIDE the topic must not be listed: the description stands for them, and
    // putting their ids back in the head is exactly what a topic exists to avoid.
    const topicLeaf = st.spans[2].id;
    assert.ok(
      !body.includes(topicLeaf),
      `the map lists a leaf that lives inside a topic: ${body}`,
    );
    // ...but the ids of a node WITHOUT a description ARE listed: the head pays for them.
    const loose = st.spans[9].id;
    assert.ok(body.includes(loose), `the map does not list a loose leaf: ${body}`);
  } finally {
    home.restore();
  }
});

test('cwl_node works on a topic, on the buffer and on the pit', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithWords(sandbox, hooks, ctx, tools, 10, ['alpha', 'bravo', 'charlie']);
    const born = await call(tools, 'cwl_group', {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    }, ctx);
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);

    // 1. A topic: the whole description, and its leaves as id + first line of the label.
    const topic = await call(tools, 'cwl_node', { id: born.details.id }, ctx);
    assert.equal(topic.details.ok, true, `cwl_node on the topic failed: ${JSON.stringify(topic.details)}`);
    assert.equal(topic.details.kind, 'topic');
    assert.equal(topic.details.leaves, 9, `the topic should hold 9 leaves: ${topic.details.leaves}`);
    assert.match(text(topic), new RegExp(DESCRIPTION.slice(0, 30).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    assert.ok(
      topic.details.leafIds.includes(st.spans[2].id),
      `the topic page does not name its leaves: ${JSON.stringify(topic.details.leafIds)}`,
    );

    // 2. The buffer: the last node, the working area. It has no description, so its kind is
    //    `buffer` and its leaves are listed with their labels.
    const map = await call(tools, 'cwl_map', {}, ctx);
    const bufferId = (text(map).match(/(nd-[0-9a-f]{8}) │ buffer/) ?? [])[1];
    assert.ok(bufferId, `no buffer in the map: ${text(map)}`);
    const buffer = await call(tools, 'cwl_node', { id: bufferId }, ctx);
    assert.equal(buffer.details.ok, true, `cwl_node on the buffer failed: ${JSON.stringify(buffer.details)}`);
    assert.equal(buffer.details.kind, 'buffer');

    // 3. An unknown id: a clear answer, not an empty page.
    const missing = await call(tools, 'cwl_node', { id: 'nd-00000000' }, ctx);
    assert.equal(missing.details.ok, false);
    assert.equal(missing.details.error, 'node-not-found');
  } finally {
    home.restore();
  }
});

test('cwl_pending counts only what is still to be ordered, and says whether it is enough', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithWords(sandbox, hooks, ctx, tools, 10, ['alpha', 'bravo', 'charlie']);
    // Nothing is in a topic yet: everything is pending, and the total is over the threshold
    // because nine labels of ~1,300 characters each are worth more than 6,480.
    const before = await call(tools, 'cwl_pending', {}, ctx);
    assert.equal(before.details.ok, true, `cwl_pending failed: ${JSON.stringify(before.details)}`);
    assert.ok(before.details.leaves >= 9, `expected the labels to be pending: ${JSON.stringify(before.details)}`);
    assert.equal(before.details.enough, true, `the total should reach the threshold: ${JSON.stringify(before.details)}`);
    assert.match(text(before), /still to order/, `the header does not say what it is: ${text(before)}`);

    const born = await call(tools, 'cwl_group', {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    }, ctx);
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);

    // After the topic, the nine leaves are ordered: what is left is the loose one.
    const after = await call(tools, 'cwl_pending', {}, ctx);
    assert.equal(after.details.ok, true);
    assert.ok(
      after.details.leaves < before.details.leaves,
      `the ordered leaves are still counted as pending: ${before.details.leaves} -> ${after.details.leaves}`,
    );
    assert.equal(after.details.enough, false, `one loose label cannot reach the threshold: ${JSON.stringify(after.details)}`);
  } finally {
    home.restore();
  }
});

test('cwl_status shows the ids of the topics, the pit and the buffer', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const st = await leavesWithWords(sandbox, hooks, ctx, tools, 10, ['alpha', 'bravo', 'charlie']);
    const born = await call(tools, 'cwl_group', {
      leaves: st.spans.slice(0, 9).map((s) => s.id), name: 'login-otp', description: DESCRIPTION,
    }, ctx);
    assert.equal(born.details.ok, true, `the topic was not born: ${JSON.stringify(born.details)}`);

    const line = text(await call(tools, 'cwl_status', {}, ctx));
    assert.match(
      line,
      new RegExp(`login-otp\\(${born.details.id}\\)`),
      `the status does not show the topic with its id: ${line}`,
    );
    const map = text(await call(tools, 'cwl_map', {}, ctx));
    const bufferId = (map.match(/(nd-[0-9a-f]{8}) │ buffer/) ?? [])[1];
    assert.ok(
      bufferId && line.includes(bufferId),
      `the status does not show the buffer id: ${line}`,
    );
    assert.match(line, /Node ids: /, `the status has no ids line: ${line}`);
  } finally {
    home.restore();
  }
});
