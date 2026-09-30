/**
 * The level-1 NODE: a container of leaves, and nothing more — for now.
 *
 * WHAT IT IS FOR, AND WHY IT IS NOT VISIBLE YET. The node does not change what stands in
 * the context: an absorbed leaf shows its micro wherever it is, and the node just
 * GROUPS it with the others (5 stay loose, the oldest ones enter).
 * What the node enables is the next step: when the nodes are three, the two
 * oldest collapse into an OLD NODE and 60 micros (18k tokens) become ONE
 * big summary (300-500t). There lies the saving; here lie the foundations.
 *
 * THE INVARIANTS THIS TEST FIXES (they are the ones the old node will inherit):
 *  1. every leaf lives in AT MOST ONE node: if a leaf ended up in two places, the
 *     content would be counted twice and the promise "nothing is lost"
 *     would become "nothing is lost, but something is paid twice";
 *  2. the leaves of a node are the OLDEST beyond the 5 loose ones, and in
 *     chronological order: the order of the top is the order of the history;
 *  3. a leaf that leaves the state also leaves the node, or the node would tell about
 *     material that no longer exists;
 *  4. an old leaf WITHOUT a micro is not absorbed and is DECLARED: its
 *     body is still in the context, and it is a cost that is stated, not hidden.
 *
 * The test drives the hook TWICE after writing the micros: the second time is needed because
 * an idempotency defect (a leaf put back into the node at every turn) is visible only
 * at the second round. It is the direction in which the test really bites.
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
});

async function boot() {
  const sandbox = makeSandbox({ name: `node-${seq++}`, config: config() });
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

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const testo = (res) => res.content.map((c) => c.text).join('\n');

test('the oldest leaves enter ONE node, only once and in order', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Eight leaves. Each round: the conversation grows (so a new
    // compressible region exists) and that region is compressed.
    for (let i = 1; i <= 8; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
      assert.equal(
        statoDi(sandbox).spans.length,
        i,
        `round ${i}: the leaves in the state should be ${i}, not ${statoDi(sandbox).spans.length}`,
      );
    }

    const foglie = statoDi(sandbox).spans;
    assert.equal(foglie.length, 8, 'eight leaves are needed: with fewer, the 5 loose ones cover everything and the node has no subject');

    // Micros on the two OLDEST and not on the third: so the node has 2 leaves and the
    // third is the one that "waits for a micro".
    for (const [i, t] of [[0, 'MICRO-1: the first story'], [1, 'MICRO-2: the second story']]) {
      const r = await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: t }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
    }

    // Two turns: the second is the one that unmasks an idempotency defect.
    await hook(hooks, ctx, conversazione(1, 12));
    const prima = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 12));
    const log = logDi(sandbox).slice(prima);

    // The node is NOT read from the file: it is DERIVED (refreshNodes rebuilds it from
    // leaves and micros at every turn), so the only honest way to observe it is
    // what the extension SAYS — the log — and what the tool DELIVERS.
    const rigaNodi = /NODES: (\d+) node\(s\) \[([^\]]*)\]/.exec(log);
    assert.ok(
      rigaNodi,
      `the turn does not declare the state of the nodes: without that line one does not know which leaves were grouped. Log of the turn: ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );
    assert.equal(Number(rigaNodi[1]), 1, `expected ONE node, declared ${rigaNodi[1]}: ${rigaNodi[2]}`);

    const ndId = /^(nd-[0-9a-f]{8}):(\d+)$/.exec(rigaNodi[2].trim());
    assert.ok(ndId, `the nodes line does not name the node and its size: "${rigaNodi[2]}"`);
    assert.equal(
      Number(ndId[2]),
      2,
      `the node must hold the two oldest leaves (the only old ones with a micro), it declares ${ndId[2]}`,
    );

    // The page of the node: the micros of its leaves, with the ids that open them.
    const aperto = await tools.get('cwl_open').execute('t', { id: ndId[1] }, undefined, undefined, ctx);
    assert.equal(aperto.details.ok, true, `the node does not open: ${JSON.stringify(aperto.details)}`);
    assert.equal(aperto.details.leaves, 2, `the page of the node declares ${aperto.details.leaves} leaves instead of 2`);

    // 1. The two leaves of the node are EXACTLY the two oldest, in
    //    chronological order, and none appears twice: if a leaf were in two
    //    places the content would be counted twice.
    const idsPagina = [...testo(aperto).matchAll(/(sp-[0-9a-f]{8})/g)].map((m) => m[1]);
    assert.deepEqual(
      idsPagina,
      [foglie[0].id, foglie[1].id],
      `the page of the node must name the two oldest leaves in order: ${JSON.stringify(idsPagina)}`,
    );
    assert.ok(
      testo(aperto).includes('MICRO-1') && testo(aperto).includes('MICRO-2'),
      `the micros are not on the page: ${testo(aperto).slice(0, 160)}`,
    );

    // 4. The old leaf without a micro is DECLARED, not hidden.
    assert.match(
      log,
      /NODES: .*waiting for a micro/,
      'the old leaf without a micro is not declared: its body is still in the context and nobody says so',
    );

    // 5. The way back also passes THROUGH the node: by removing the micro, the leaf leaves.
    const via = await tools.get('cwl_micro').execute('t', { id: foglie[0].id, text: '' }, undefined, undefined, ctx);
    assert.equal(via.details.ok, true, `removing the micro failed: ${JSON.stringify(via.details)}`);
    // The offset is taken HERE, not earlier: from `prima` onward there are TWO turns, and
    // `exec` would return the line of the first — which still says :2. It is the third time
    // today that this trap bites: one reads the log stretch of the turn being
    // measured, never "from point X onward".
    const primaDelRitorno = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 12));
    const dopo = /NODES: (\d+) node\(s\) \[([^\]]*)\]/.exec(logDi(sandbox).slice(primaDelRitorno));
    assert.ok(dopo, 'after removing the micro the turn no longer declares the state of the nodes');
    assert.equal(
      dopo[2].trim(),
      `${ndId[1]}:1`,
      `by removing the micro the leaf stayed in the node: ${dopo[2]} — the body went back into the context, so the node would count it twice`,
    );
  } finally {
    home.restore();
  }
});
