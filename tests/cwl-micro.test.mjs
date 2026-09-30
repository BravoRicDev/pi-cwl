/**
 * `cwl_micro`: il corpo esce dal contesto, il micro prende il suo posto, e il
 * corpo resta INTERO a una chiamata di distanza.
 *
 * E' la leva da cui e' nato tutto il progetto. MISURATO in una sessione vera
 * (commit de9672a):
 *
 *   SPANS content: 52535t inside the spans = 52027t of summaries + 508t of user
 *   turns + 0t of other roles
 *
 * 52.027 token su 114.327 erano i riassunti che l'estensione aveva scritto LEI
 * (30 foglie, media 1.707t), e nessuna via poteva toccarli: `keptInsideSpan` tiene
 * `custom` e l'applier dell'evacuazione protegge `custom`. L'estensione sapeva
 * SCRIVERE un riassunto e non sapeva ASSORBIRNE uno vecchio: ogni compressione
 * aggiungeva un racconto, nessuna leva ne ritirava mai uno.
 *
 * LE DUE COSE CHE QUESTO TEST DIFENDE:
 *  1. assorbire fa DIMINUIRE il contesto: il corpo esce, il micro entra. Se il
 *     corpo restasse, l'assorbimento non farebbe nulla;
 *  2. il micro NON tocca il corpo. Se lo scrivesse dentro, `cwl_open` tornerebbe
 *     corto e la promessa "non si perde niente" sarebbe falsa con una riga: e'
 *     l'unico modo di rompere questo design che non si vede dal contesto.
 *
 * C'e' anche la via di ritorno: un testo VUOTO rimuove il micro e rimette il
 * corpo intero nel contesto. Un assorbimento che nessuno puo' annullare e' una
 * porta a senso unico, e questa ce l'ha.
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
  const sandbox = makeSandbox({ name: `micro-${seq++}`, config: config() });
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

const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'lo stato della sessione non e\' stato creato');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

/** Cio' che il provider riceverebbe: i blocchi iniettati al posto dei compressi. */
const inContesto = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-compressed').map((m) => String(m.content)).join('\n');

const testo = (res) => res.content.map((c) => c.text).join('\n');

test('il micro sostituisce il corpo nel contesto senza distruggerlo', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await hook(hooks, ctx, conversazione());

    const corpo = 'RIASSUNTO-INTEGRALE ' + 'x'.repeat(8000) + ' FINE-DEL-CORPO';
    const micro = 'MICRO-BREVE: i punti che contano davvero.';
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: corpo }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `lo span non e' stato creato: ${JSON.stringify(comp.details)}`);
    const sp = statoDi(sandbox).spans[0];
    assert.ok(sp && sp.id, `lo span non ha un id: ${JSON.stringify(sp)}`);

    // Prima dell'assorbimento il corpo e' nel contesto, per intero.
    const prima = await hook(hooks, ctx, conversazione());
    assert.ok(
      inContesto(prima).includes('FINE-DEL-CORPO'),
      'la premessa del test non regge: il corpo non era nel contesto nemmeno prima dell\'assorbimento',
    );

    const ass = await tools.get('cwl_micro').execute('t', { id: sp.id, text: micro }, undefined, undefined, ctx);
    assert.equal(ass.details.ok, true, `l'assorbimento e' fallito: ${JSON.stringify(ass.details)}`);

    const dopo = await hook(hooks, ctx, conversazione());
    const contesto = inContesto(dopo);

    // 1. Il corpo e' USCITO dal contesto e il micro e' entrato: e' tutto il punto.
    assert.ok(contesto.includes(micro), `il micro non e' nel contesto: ${contesto.slice(0, 160)}`);
    assert.ok(
      !contesto.includes('FINE-DEL-CORPO'),
      'il corpo e\' ancora nel contesto: l\'assorbimento non ha liberato niente, ' +
        'quindi l\'estensione continua a pagare per la stessa storia',
    );

    // 2. Il corpo NON e' stato distrutto: `cwl_open` lo restituisce tutto.
    const riaperto = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    assert.equal(riaperto.details.ok, true, `la foglia non si riapre piu': ${JSON.stringify(riaperto.details)}`);
    assert.ok(
      testo(riaperto).includes(corpo),
      `il micro ha mangiato il corpo: cwl_open risponde con ${testo(riaperto).length} caratteri, il corpo ne ha ${corpo.length}. ` +
        'La promessa "non si perde niente" sarebbe falsa, e dal contesto non si vedrebbe.',
    );

    // 3. Un micro PIU' LUNGO del corpo e' dichiarato per quello che e'.
    const lungo = await tools.get('cwl_micro').execute(
      't', { id: sp.id, text: 'y'.repeat(corpo.length + 100) }, undefined, undefined, ctx,
    );
    assert.equal(
      lungo.details.shorter,
      false,
      'un micro piu\' lungo del corpo che sostituisce e\' passato come un risparmio: il contesto non si riduce, e l\'agente non lo saprebbe',
    );

    // 4. La via di ritorno: un testo vuoto rimette il corpo intero nel contesto.
    const via = await tools.get('cwl_micro').execute('t', { id: sp.id, text: '' }, undefined, undefined, ctx);
    assert.equal(via.details.ok, true, `la rimozione del micro e' fallita: ${JSON.stringify(via.details)}`);
    const ritorno = inContesto(await hook(hooks, ctx, conversazione()));
    assert.ok(
      ritorno.includes('FINE-DEL-CORPO'),
      'togliendo il micro il corpo non e\' tornato nel contesto: l\'assorbimento era una porta a senso unico',
    );
  } finally {
    home.restore();
  }
});
