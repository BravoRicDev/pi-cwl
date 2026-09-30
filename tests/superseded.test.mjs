/**
 * The synthesis a merge REPLACES must not vanish.
 *
 * `cwl_old` is the only destructive tool of the set: it OVERWRITES the pit's synthesis
 * instead of extending it. The operator measured the cost in a real session — a rewrite that
 * does not carry the previous text forward erases the only copy the archive has. The remedy
 * is one page per replaced synthesis, addressed `<pit id>.s1`, `.s2`, ... from the newest,
 * listed on the pit page with their size and readable in full on demand.
 *
 * THE THREE DIRECTIONS IN WHICH THE TEST MUST DIE:
 *  1. the merge overwrites the synthesis and the previous text is gone -> nothing can be
 *     read back and nothing points to it;
 *  2. the pit page grows with every replaced synthesis, instead of listing them: the page
 *     is the hub of the archive and has to stay a page;
 *  3. the list grows without bound, so the state file pays for every merge ever made.
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
  // One young node is enough to make a merge due, so each `cwl_old` below replaces the
  // synthesis of the pit instead of waiting for the index to fill.
  mergeNodesAt: 1,
  // The size guards are lowered on purpose: they are measured against labels of ~1,200
  // characters and this fixture writes labels like `BLOCK-1`. The guard has its own test.
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot() {
  const sandbox = makeSandbox({ name: `superseded-${seq++}`, config: config() });
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

test('the synthesis a merge replaced stays readable, and the list stays bounded', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Ten leaves, two per node: five merges are available below.
    for (let i = 1; i <= 10; i++) {
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

    // Merge 1: SYNTHESIS-1 becomes the pit's synthesis; nothing was replaced yet.
    await hook(hooks, ctx, conversation(1, 14));
    const first = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS-1: the first stories' }, undefined, undefined, ctx);
    assert.equal(first.details.ok, true, `the first merge was refused: ${JSON.stringify(first.details)}`);
    const pit = first.details.id;
    assert.equal(
      stateOf(sandbox).oldNode.superseded,
      undefined,
      'the first merge replaced nothing, so nothing has to be kept',
    );

    // Merge 2: SYNTHESIS-1 is REPLACED. It must survive somewhere, and the pit page must say where.
    await hook(hooks, ctx, conversation(1, 14));
    const second = await tools.get('cwl_old').execute('t', { text: 'SYNTHESIS-2: and the stories after' }, undefined, undefined, ctx);
    assert.equal(second.details.ok, true, `the second merge was refused: ${JSON.stringify(second.details)}`);
    assert.equal(
      stateOf(sandbox).oldNode.superseded?.[0],
      'SYNTHESIS-1: the first stories',
      'the synthesis the merge replaced was not kept: cwl_old overwrites, so it is gone',
    );

    const page = text(await tools.get('cwl_open').execute('t', { id: pit }, undefined, undefined, ctx));
    assert.ok(page.includes('SYNTHESIS-2'), `the pit page must carry the CURRENT synthesis: ${page.slice(0, 200)}`);
    assert.ok(page.includes(`${pit}.s1`), `the pit page must point to the synthesis it replaced: ${page.slice(0, 400)}`);
    assert.ok(
      !page.includes('SYNTHESIS-1'),
      'the replaced text is DUMPED into the pit page: the page lists what it kept, it does not carry it',
    );

    const old = text(await tools.get('cwl_open').execute('t', { id: `${pit}.s1` }, undefined, undefined, ctx));
    assert.ok(
      old.includes('SYNTHESIS-1: the first stories'),
      `the replaced synthesis is not readable: ${old.slice(0, 200)}`,
    );
    assert.ok(old.includes(`${pit}.s1`), 'the page does not name itself');

    // And the list is BOUNDED: no merge ever grows it past the ceiling.
    for (let round = 3; round <= 8; round++) {
      await hook(hooks, ctx, conversation(1, round + 14));
      const res = await tools.get('cwl_old').execute('t', { text: `SYNTHESIS-${round}` }, undefined, undefined, ctx);
      if (!res.details.ok) continue; // no young node left to absorb: nothing was replaced
      const kept = stateOf(sandbox).oldNode.superseded ?? [];
      assert.ok(
        kept.length <= 3,
        `the superseded list grew to ${kept.length}: the state file would pay for every merge ever made`,
      );
      assert.equal(kept[0], `SYNTHESIS-${round - 1}`, 'the newest replaced synthesis must come first');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});
