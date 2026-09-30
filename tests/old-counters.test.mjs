/**
 * THE COUNTERS: the old-node page shows the leaves CONSULTED, not the most
 * recent ones.
 *
 * THE RULE IS THE OPERATOR'S, and it is worth restating because it is counterintuitive:
 * *"the 30 most recently absorbed ones do not interest me: 'consulted often', which
 * must anyway remain visible inside the old node. The information from the
 * most recent leaves, if it was used, still leaves a trail of
 * reasoning in the 5 youngest ones and in the untouchable part: it has no need to
 * be there too. It is the information that LOOKS STALE that must be
 * visible inside the old node without going to look for it in the absorbed young node."*
 *
 * WHY THE COUNTER IS THE RIGHT SIGNAL, even if it looks leaky. It records only
 * EXPLICIT opens: a leaf used while it was fresh leaves no trace, because
 * its body was already in the context and nobody had to open it. It looks like a hole and it is
 * the right behaviour — a leaf that is useful while fresh needs no help. The
 * counter measures exactly the other case: leaves RE-CONSULTED AFTER they
 * aged. The hole and the case that needs no help coincide.
 *
 * AND THE COUNTERS ARE NOT A POLICY: they order a page and that is it. They delete
 * nothing, because "never opened" does not mean "useless" (a leaf may have been
 * used perfectly well while fresh). In the fixture the OLDEST leaf is opened twice
 * and the most recent one zero times: without the counters the date order would put the
 * most recent first, and that is exactly the direction in which the test must die.
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
  const sandbox = makeSandbox({ name: `counters-${seq++}`, config: config() });
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
    out.push({ role: 'user', content: `prompt ${i} content. ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answers ${i} content is ` + 'A'.repeat(200) });
  }
  return out;
};

const testo = (res) => res.content.map((c) => c.text).join('\n');
const apri = (tools, ctx, id) => tools.get('cwl_open').execute('t', { id }, undefined, undefined, ctx);

test('the old-node page orders the leaves by consultations, not by date', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const foglie = statoDi(sandbox).spans;
    for (let i = 0; i < 5; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro on leaf ${i + 1} failed: ${JSON.stringify(r.details)}`);
    }
    await hook(hooks, ctx, conversazione(1, 12));
    const acc = await tools.get('cwl_old').execute('t', { text: 'SUMMARY-ALL-1' }, undefined, undefined, ctx);
    assert.equal(acc.details.ok, true, `the well was not formed: ${JSON.stringify(acc.details)}`);
    const pozzo = acc.details.id;

    // 1. EMPTY history: the page still lists the leaves of the well (date order,
    //    the sensible fallback) and declares that none was ever opened.
    const fredda = await apri(tools, ctx, pozzo);
    assert.equal(fredda.details.ok, true, `the well does not open: ${JSON.stringify(fredda.details)}`);
    assert.equal(fredda.details.opened, 0, `with empty history no leaf was opened, but it declares ${fredda.details.opened}`);
    assert.equal(fredda.details.leaves, 2, `the well holds 2 leaves, but it declares ${fredda.details.leaves}`);
    assert.match(testo(fredda), /opened 0x/, 'the page does not declare how many times each leaf was opened');

    // 2. The OLDEST leaf is opened twice.
    for (let k = 0; k < 2; k++) await apri(tools, ctx, foglie[0].id);

    // The counter lives with the leaf in the state: that is what makes it survive
    // /reload, and it is the only thing the test must see to believe it.
    const dopoTurno = statoDi(sandbox).spans.find((s) => s.id === foglie[0].id);
    assert.equal(dopoTurno.opens, 2, `the leaf opened twice counts ${dopoTurno.opens}: the counter is not in the state, so it does not survive a restart`);
    assert.ok(dopoTurno.lastOpen, 'the instant of the last open is missing');

    // 3. Now the page must put it FIRST: that is the operator's rule.
    const primaDellApertura = logDi(sandbox).length;
    const calda = await apri(tools, ctx, pozzo);
    const log = logDi(sandbox).slice(primaDellApertura);
    const righe = testo(calda).split('\n').filter((l) => l.startsWith('- sp-'));
    assert.ok(righe.length >= 2, `the page does not list the leaves: ${testo(calda).slice(0, 200)}`);
    assert.ok(
      righe[0].includes(foglie[0].id),
      `the first leaf listed is not the most consulted one: the page orders by date, not by consultations. First line: ${righe[0]}`,
    );
    assert.ok(righe[0].includes('opened 2x'), `the line of the most consulted leaf does not declare the 2 opens: ${righe[0]}`);
    assert.equal(calda.details.opened, 1, `the never-opened leaves are 1 (out of 2), but it declares ${calda.details.opened}`);
    assert.match(
      log,
      /most opened 2x/,
      `the log does not measure the use of the well (how many times the most opened leaf was opened): ${log.trim().split('\n').slice(-2).join(' | ')}`,
    );
  } finally {
    home.restore();
  }
});
