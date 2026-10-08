/**
 * SOTTO SOGLIA LE EVICTION E LA MANUTENZIONE NON DEVONO FARE NULLA.
 *
 * Regola stabilita dall'operatore: "le EVICTION non devono fare assolutamente nulla
 * se siamo sotto soglia, vedo che rompono la cache. Devono agire solo quando la soglia
 * viene superata per dare più respiro ed evitare overflow non hanno nessun beneficio
 * prima di quel momento, invalidano la cache per niente."
 *
 * Questo test verifica che:
 *  1. Sotto soglia, anche se ci sono nodi giovani pronti per il merge (young >=
 *     mergeNodesAt), foglie in attesa di micro, o un buffer pronto per un topic, CWL NON
 *     inietta alcun messaggio nel contesto: la prefix cache resta intatta.
 *  2. E che SOPRA soglia NON ne inietta nessuno comunque. Questa era la meta' opposta del
 *     test fino alla decisione dell'operatore ("togliere qualsiasi avviso che chiede
 *     all'agente di comprimere"): la demand non e' piu' un canale, sopra o sotto. Cio' che
 *     resta e' il LOG, ed e' quello che il test pretende adesso — perche' un meccanismo che
 *     smette di parlare in silenzio e' un meccanismo che nessuno puo' diagnosticare.
 *  3. Che il gate non scriva piu' nel contesto nemmeno il proprio messaggio custom.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (budget = 600) => ({
  tokenBudget: budget,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 2,
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot(budget = 600) {
  const sandbox = makeSandbox({ name: `quiet-${seq++}`, config: config(budget) });
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
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const demandRequest = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');

// Not exported by `_helpers.mjs`: every test that reads the log defines it in house.
const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const isGate = (msgs) => msgs.filter((m) => m && m.customType === 'cwl-budget-gate').length;

test('sopra E sotto soglia: zero richieste nel contesto (preserva prefix cache)', async () => {
  // 1. Creiamo nodi giovani in una sessione
  const { sandbox, home, tools, hooks, ctx } = await boot(600);
  try {
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: leaf failed: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox).spans;
    for (let i = 0; i < leaves.length; i++) {
      await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
    }

    // SOPRA SOGLIA: conversation(1, 12) genera ~1200 token > 300 trigger.
    // La demand NON entra nel contesto nemmeno qui, e nemmeno il messaggio custom del gate.
    // La CONDIZIONE pero' deve restare detta nel LOG: e' l'unico canale rimasto, ed e' la
    // meta' silenziosa di questa storia che una volta e' costata una sessione.
    const logBefore = logOf(sandbox).length;
    const overMsgs = await hook(hooks, ctx, conversation(1, 12));
    const overText = demandRequest(overMsgs);
    assert.equal(
      overText.length,
      0,
      `sopra soglia nessuna richiesta deve entrare nel contesto (trovata: ${overText.slice(0, 200)})`,
    );
    assert.equal(isGate(overMsgs), 0, 'sopra soglia il gate non deve iniettare il proprio messaggio custom');

    const overRows = logOf(sandbox).slice(logBefore);
    assert.match(
      overRows,
      /OLD NODE due: 2\+ young nodes/,
      `la condizione di merge deve restare detta nel LOG, col numero di nodi: ${overRows.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      overRows,
      /LOG ONLY now/,
      `la riga di log deve dire che il log e' l'unico canale rimasto: ${overRows.trim().split('\n').slice(-4).join(' | ')}`,
    );

    // Ora riavviamo l'estensione nello stesso sandbox MA con budget alto (100_000, trigger 50_000).
    // Lo stato persiste sul disco con i suoi nodi giovani pronti per il merge.
    const cfgPath = path.join(sandbox.dir, '.pi', 'cwl', 'config.json');
    const newCfg = { ...config(100_000) };
    fs.writeFileSync(cfgPath, JSON.stringify(newCfg, null, 2), 'utf8');

    const ext2 = await bootExtension(sandbox, { name: 'ext2' });
    const ctx2 = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    await ext2.hooks.get('session_start')({}, ctx2);

    // SOTTO SOGLIA: la conversazione (12 turni = ~1200 token) è ben sotto il trigger di 50_000.
    // CWL NON deve iniettare alcuna demand di merge/micro nel contesto!
    const underMsgs = await hook(ext2.hooks, ctx2, conversation(1, 12));
    const underText = demandRequest(underMsgs);
    assert.equal(
      underText.length,
      0,
      `sotto soglia non deve essere iniettata alcuna demand (trovata: ${underText.slice(0, 200)})`,
    );

    const demandMsgs = underMsgs.filter((m) => m && m.customType === 'cwl-demand');
    assert.equal(demandMsgs.length, 0, 'nessun custom message cwl-demand deve essere presente sotto soglia');
    assert.equal(isGate(underMsgs), 0, 'sotto soglia nemmeno il gate deve scrivere nel contesto');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
