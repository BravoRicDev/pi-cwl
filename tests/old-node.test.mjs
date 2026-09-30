/**
 * The OLD NODE: where the index finally SAVES.
 *
 * It is the step that justifies the whole project. MEASURED in a real session
 * (commit de9672a):
 *
 *   SPANS content: 52535t inside the spans = 52027t of summaries + 508t of user
 *   turns + 0t of other roles
 *
 * 52,027 tokens out of 114,327 were the summaries the extension had written ITSELF, and
 * no lever could touch them: `keptInsideSpan` keeps `custom`, the eviction
 * applier protects `custom`. It knew how to write a summary and did not know how to
 * absorb an old one.
 *
 * Here 4 leaves (4 micros, ~300t each in the real design) leave the context and in their
 * place ONE synthesis enters. The leaves are not lost: they stay in the state, and
 * `cwl_open` reopens them whole — it is the promise of the design, and it must be tried in the same
 * test in which the saving is measured, otherwise "saving" could mean "throwing away".
 *
 * THE KNOBS. The test does not build 90 leaves to fill 3 nodes of 30: it uses
 * `looseLeaves: 1`, `nodeCapacity: 2`, `mergeNodesAt: 2`, which exist as config
 * exactly for this (and because the shape of the index is a preference
 * of the operator, like `protectedTurns`).
 *
 * THE FOUR DIRECTIONS IN WHICH THE TEST MUST DIE:
 *  1. the pit's leaves are still injected one by one -> no saving;
 *  2. the synthesis does not enter the context -> the leaves vanish from the head and NOTHING
 *     stands for them: the thread of the conversation is cut, which is worse than not
 *     compressing;
 *  3. `cwl_old` also merges the YOUNGEST node -> the present ends up in the pit;
 *  4. `cwl_old` merges when it is not due -> the synthesis of nothing.
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
  mergeNodesAt: 2,
});

async function boot() {
  const sandbox = makeSandbox({ name: `old-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

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

const inContext = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-compressed').map((m) => String(m.content)).join('\n');

const text = (res) => res.content.map((c) => c.text).join('\n');

test('the old node replaces the micros with the synthesis, and does not lose the leaves', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // 0. Not due: no node. The tool must REFUSE, not merge nothing.
    const tooEarly = await tools.get('cwl_old').execute('t', { text: 'SINTESI-PRECOCE' }, undefined, undefined, ctx);
    assert.equal(
      tooEarly.details.ok,
      false,
      'cwl_old merged when there was nothing to merge yet: the synthesis would describe the void',
    );

    // Five leaves, each with its micro. With looseLeaves 1 and nodeCapacity 2:
    // the newest stays loose, the other four form TWO nodes.
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox).spans;
    assert.equal(leaves.length, 5, `expected 5 leaves, ${leaves.length} in the state`);
    for (let i = 0; i < 5; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
    }

    // One turn to form the nodes. The merge is DUE and must be declared.
    const beforePit = logOf(sandbox).length;
    await hook(hooks, ctx, conversation(1, 12));
    const log = logOf(sandbox).slice(beforePit);
    assert.match(
      log,
      /OLD NODE due/,
      `with 2 young nodes and mergeNodesAt 2 the merge is due, and no line says so: ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );

    // Before the merge the micros are ALL in the context: it is the premise.
    const before = inContext(await hook(hooks, ctx, conversation(1, 12)));
    for (let i = 1; i <= 5; i++) {
      assert.ok(before.includes(`MICRO-${i}`), `before the merge MICRO-${i} is missing: the fixture does not hold`);
    }

    // The merge: the oldest node (leaves 1 and 2) enters the pit.
    const acc = await tools.get('cwl_old').execute(
      't', { text: 'RIASSUNTONE-1: the synthesis of the first two stories' }, undefined, undefined, ctx,
    );
    assert.equal(acc.details.ok, true, `cwl_old refused when it was due: ${JSON.stringify(acc.details)}`);
    assert.equal(acc.details.nodes, 1, `it had to merge ONE node (the oldest), it declares ${acc.details.nodes}`);
    assert.equal(acc.details.leaves, 2, `it had to merge 2 leaves, it declares ${acc.details.leaves}`);
    const pit = acc.details.id;
    assert.match(String(pit), /^old-[0-9a-f]{8}$/, `unexpected pit id: ${pit}`);

    // The next turn: the synthesis is in the context, the merged micros are NOT there anymore.
    const after = inContext(await hook(hooks, ctx, conversation(1, 12)));
    assert.ok(after.includes('RIASSUNTONE-1'), 'the synthesis did not enter the context: the leaves went out and nobody stands for them');
    for (const i of [1, 2]) {
      assert.ok(
        !after.includes(`MICRO-${i}`),
        `MICRO-${i} is still in the context: the pit's leaves are still injected, so the merge freed nothing`,
      );
    }
    // And the YOUNG node no: the present does not enter the pit.
    for (const i of [3, 4, 5]) {
      assert.ok(after.includes(`MICRO-${i}`), `MICRO-${i} vanished from the context: it ended up in the pit together with the past`);
    }

    // The pit page: the synthesis, and the SHAPE of what it holds inside.
    const page = await tools.get('cwl_open').execute('t', { id: pit }, undefined, undefined, ctx);
    assert.equal(page.details.ok, true, `the pit does not open: ${JSON.stringify(page.details)}`);
    assert.equal(page.details.nodes, 1, `the pit declares ${page.details.nodes} nodes inside instead of 1`);
    assert.ok(text(page).includes('RIASSUNTONE-1'), 'the pit page does not report the synthesis');
    assert.match(
      text(page),
      /2 leaf\/leaves/,
      `the page does not declare the shape (how many leaves) of the node it holds: ${text(page).slice(0, 200)}`,
    );

    // And nothing was lost: the merged leaves reopen WHOLE.
    const reopened = await tools.get('cwl_open').execute('t', { id: leaves[0].id }, undefined, undefined, ctx);
    assert.equal(reopened.details.ok, true, `the merged leaf does not reopen: ${JSON.stringify(reopened.details)}`);
    assert.ok(
      text(reopened).includes('CORPO-1'),
      'the body of the merged leaf does not come back whole: the saving would be a throwing away, not a compressing',
    );
  } finally {
    home.restore();
  }
});
