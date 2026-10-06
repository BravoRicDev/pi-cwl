/**
 * SOTTO SOGLIA LE EVICTION E LA MANUTENZIONE NON DEVONO FARE NULLA.
 *
 * Regola stabilita dall'operatore: "le EVICTION non devono fare assolutamente nulla
 * se siamo sotto soglia, vedo che rompono la cache. Devono agire solo quando la soglia
 * viene superata per dare più respiro ed evitare overflow non hanno nessun beneficio
 * prima di quel momento, invalidano la cache per niente."
 *
 * Questo test verifica che:
 *  1. Sotto soglia (roomNeeded === false), anche se ci sono nodi giovani pronti per
 *     il merge (young >= mergeNodesAt), foglie in attesa di micro, o un buffer pronto
 *     per un topic, CWL NON inietta alcun messaggio di demand nel contesto. La prefix
 *     cache resta intatta e l'agente non viene disturbato.
 *  2. Non appena la soglia viene superata (roomNeeded === true), la demand di merge
 *     entra nel contesto per permettere all'agente di liberare spazio.
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

test('sotto soglia: zero richieste di merge o micro nel contesto (preserva prefix cache)', async () => {
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
    // La demand entra nel contesto per permettere all'agente di liberare spazio.
    const overMsgs = await hook(hooks, ctx, conversation(1, 12));
    const overText = demandRequest(overMsgs);
    assert.ok(overText.length > 0, 'sopra soglia la richiesta di merge deve entrare nel contesto');
    assert.match(overText, /cwl_old/, 'la demand deve indicare cwl_old');

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
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
