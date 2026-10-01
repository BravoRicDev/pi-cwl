/**
 * THE BODIES ON DISK — the level-0 rule and the operator's rule together.
 *
 * Level-0 (decided together): every leaf OUTSIDE the pit keeps micro + full summary in
 * RAM; a leaf absorbed into the pit has its summary moved to an append-only bodies file.
 * The operator's rule on top: MICRO + SUMMARY must ALWAYS remain — the original transcript
 * may go away. So even the leaf that DIES (its anchors left the context for good) keeps its
 * summary on disk: it is appended to the body store BEFORE the span leaves the state, and
 * `cwl_open` reads it back at the stored offset.
 *
 * What this test pins:
 *  1. the state JSON no longer carries the absorbed summaries — the bodies file does;
 *  2. `cwl_open` on an absorbed leaf returns the byte-identical body, read from disk;
 *  3. the FORK copies the bodies file, so `cwl_open` works in the forked session too;
 *  4. a DEAD leaf is pruned as before (spans goes to zero) but its summary survives on
 *     disk and `cwl_open` returns it.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const mergeConfig = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  // The same lowered guards as old-node.test.mjs: five tiny leaves form TWO nodes and the
  // merge is due at 2 — the pit exists without the real size guards, which are not what
  // this test is about.
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 2,
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot(config) {
  const sandbox = makeSandbox({ name: `bodies-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const aFile = path.join(sandbox.dir, 'sessione-A.jsonl');
  const a = sessionCtx(aFile);
  await hooks.get('session_start')({}, a);
  return { sandbox, home, tools, hooks, a, aFile };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const stateFileOf = (sandbox, sessionPath) =>
  path.join(sandbox.dir, '.pi', 'cwl', 'state',
    `${createHash('sha256').update(sessionPath).digest('hex').slice(0, 32)}.json`);
const bodiesFileOf = (sandbox, sessionPath) => stateFileOf(sandbox, sessionPath).replace(/\.json$/, '.bodies.jsonl');
const stateOf = (sandbox, sessionPath) => JSON.parse(fs.readFileSync(stateFileOf(sandbox, sessionPath), 'utf8'));
const rawOf = (sandbox, sessionPath) => fs.readFileSync(stateFileOf(sandbox, sessionPath), 'utf8');
const bodiesOf = (sandbox, sessionPath) =>
  fs.existsSync(bodiesFileOf(sandbox, sessionPath)) ? fs.readFileSync(bodiesFileOf(sandbox, sessionPath), 'utf8') : '';

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const text = (res) => res.content.map((c) => c.text).join('\n');

test('a leaf absorbed into the pit leaves the state JSON and its body opens from disk, byte-identical', async () => {
  const { sandbox, home, tools, hooks, a, aFile } = await boot(mergeConfig());
  try {
    // Five leaves, each with its micro (the old-node pattern). With looseLeaves 1 and
    // nodeCapacity 2 the oldest two form the node that the merge will absorb.
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, a, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, a,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox, aFile).spans;
    assert.equal(leaves.length, 5, 'expected 5 leaves');
    const firstId = leaves[0].id; // BLOCK-1
    const firstBody = leaves[0].summary;
    const firstAt = new Date(leaves[0].at).toISOString().slice(0, 16).replace('T', ' ');
    assert.ok(firstBody.startsWith('BLOCK-1 '), 'the fixture is not the one expected');
    for (let i = 0; i < 5; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, a);
      assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
    }

    // One turn to form the nodes, then the merge: the oldest node (BLOCK-1, BLOCK-2) enters
    // the pit. cwl_old saves at once, and the save moves the pit's bodies to disk.
    await hook(hooks, a, conversation(1, 12));
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the oldest' }, undefined, undefined, a);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);

    // 1. Level-0 rule, measured on the files: the absorbed summaries LEFT the state JSON,
    // the ones outside the pit stayed, and the bodies file carries what left.
    const raw = rawOf(sandbox, aFile);
    assert.ok(!raw.includes('BLOCK-1 '), 'the absorbed summary is still in the state JSON');
    assert.ok(!raw.includes('BLOCK-2 '), 'the absorbed summary is still in the state JSON');
    assert.ok(raw.includes('BLOCK-3 '), 'a leaf OUTSIDE the pit lost its summary from the state');
    const bodies = bodiesOf(sandbox, aFile);
    assert.ok(bodies.includes('BLOCK-1 ') && bodies.includes('BLOCK-2 '), 'the bodies file does not carry the absorbed summaries');

    // 2. cwl_open on the absorbed leaf: the body comes back whole, from disk at its offset.
    const opened = await tools.get('cwl_open').execute('t', { id: firstId }, undefined, undefined, a);
    assert.equal(opened.details.ok, true, `opening the absorbed leaf failed: ${JSON.stringify(opened.details)}`);
    assert.equal(text(opened), `[CWL leaf ${firstId} — compressed ${firstAt}, ~${opened.details.tokens} tokens. The WHOLE body follows; nothing is truncated.]\n\n${firstBody}`);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the fork copies the bodies file, so an absorbed leaf opens in the forked session too', async () => {
  const { sandbox, home, tools, hooks, a, aFile } = await boot(mergeConfig());
  try {
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, a, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, a,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox, aFile).spans;
    const firstId = leaves[0].id;
    for (let i = 0; i < 5; i++) {
      await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, a);
    }
    await hook(hooks, a, conversation(1, 12));
    const merged = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS of the oldest' }, undefined, undefined, a);
    assert.equal(merged.details.ok, true, `the merge was refused: ${JSON.stringify(merged.details)}`);
    const name = stateOf(sandbox, aFile).name;
    assert.ok(name, 'the source memory has no name');

    // The fork into a SECOND session of the same sandbox.
    const bFile = path.join(sandbox.dir, 'sessione-B.jsonl');
    const b = sessionCtx(bFile);
    await hooks.get('session_start')({}, b);
    const adopted = await tools.get('cwl_adopt').execute('t', { from: name, as: 'ramo' }, undefined, undefined, b);
    assert.equal(adopted.details.ok, true, `the fork failed: ${JSON.stringify(adopted.details)}`);

    // The fork owns its bodies file, and the offset index is valid there: the absorbed leaf
    // opens in the forked session, byte-identical.
    const openedB = await tools.get('cwl_open').execute('t', { id: firstId }, undefined, undefined, b);
    assert.equal(openedB.details.ok, true, `opening the leaf in the fork failed: ${JSON.stringify(openedB.details)}`);
    assert.ok(text(openedB).includes('BLOCK-1 '), 'the fork did not copy the bodies file');
    // And the fork is a COPY: the source bodies file still exists and is untouched.
    assert.ok(bodiesOf(sandbox, aFile).includes('BLOCK-1 '), 'the source bodies file was moved or emptied');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a DEAD leaf is pruned as before, but its summary and micro survive on disk and open', async () => {
  const { sandbox, home, tools, hooks, a, aFile } = await boot({
    tokenBudget: 600,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
    showWidget: false,
    debug: true,
  });
  try {
    const turn = (i) => ([
      { role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) },
      { role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) },
    ]);
    const list = [];
    for (let i = 1; i <= 6; i++) list.push(...turn(i));
    await hooks.get('context')({ messages: list }, a);
    const out = await tools.get('cwl_compress_range').execute(
      't', { summary: 'summary of the first half', micro: 'MICRO: the first half.' }, undefined, undefined, a,
    );
    assert.equal(out.details.ok, true, `rejected: ${JSON.stringify(out.details)}`);
    const id = stateOf(sandbox, aFile).spans[0].id;

    // A native compaction takes the endpoints: the leaf can never apply again. It is pruned
    // — spans goes to zero, the old contract — but the body store keeps its summary + micro.
    await hooks.get('context')({ messages: list.slice(-2) }, a);
    const stat = await tools.get('cwl_status').execute('t', {}, undefined, undefined, a);
    assert.equal(stat.details.spans, 0, 'the dead leaf must still leave the state');
    const bodies = bodiesOf(sandbox, aFile);
    assert.ok(bodies.includes('summary of the first half'), 'the dead summary was not appended to the bodies file');
    assert.ok(bodies.includes('MICRO: the first half.'), 'the dead micro was not preserved');

    // And it OPENS: the summary is the keeper now, the original transcript is optional.
    const opened = await tools.get('cwl_open').execute('t', { id }, undefined, undefined, a);
    assert.equal(opened.details.ok, true, `the dead leaf does not open: ${JSON.stringify(opened.details)}`);
    assert.ok(text(opened).includes('summary of the first half'), 'the opened body is not the preserved summary');
  } finally { home.restore(); sandbox.cleanup(); }
});
