/**
 * Il NODO VECCHIO: dove l'indice finalmente RISPARMIA.
 *
 * E' il passo che giustifica tutto il progetto. MISURATO in una sessione vera
 * (commit de9672a):
 *
 *   SPANS content: 52535t inside the spans = 52027t of summaries + 508t of user
 *   turns + 0t of other roles
 *
 * 52.027 token su 114.327 erano i riassunti che l'estensione aveva scritto LEI, e
 * nessuna leva poteva toccarli: `keptInsideSpan` tiene `custom`, l'applier
 * dell'evacuazione protegge `custom`. Sapeva scrivere un riassunto e non sapeva
 * assorbirne uno vecchio.
 *
 * Qui 4 foglie (4 micro, ~300t l'uno nel design vero) escono dal contesto e al loro
 * posto entra UNA sintesi. Le foglie non si perdono: restano nello stato, e
 * `cwl_open` le riapre intere — e' la promessa del design, e va provata nello stesso
 * test in cui si misura il risparmio, altrimenti "risparmiare" potrebbe voler dire
 * "buttare".
 *
 * LE MANOPOLE. Il test non costruisce 90 foglie per riempire 3 nodi da 30: usa
 * `looseLeaves: 1`, `nodeCapacity: 2`, `mergeNodesAt: 2`, che esistono come config
 * proprio per questo (e perche' la forma dell'indice e' una preferenza
 * dell'operatore, come `protectedTurns`).
 *
 * LE QUATTRO DIREZIONI IN CUI IL TEST DEVE MORIRE:
 *  1. le foglie del pozzo vengono ancora iniettate una per una -> nessun risparmio;
 *  2. la sintesi non entra nel contesto -> le foglie spariscono dalla testa e NIENTE
 *     sta per loro: il filo del discorso viene tagliato, che e' peggio del non
 *     comprimere;
 *  3. `cwl_old` accorpa anche il nodo PIU' GIOVANE -> il presente finisce nel pozzo;
 *  4. `cwl_old` accorpa quando non e' dovuto -> la sintesi di niente.
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
  const sandbox = makeSandbox({ name: `vecchio-${seq++}`, config: config() });
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
  assert.ok(file, 'lo stato della sessione non e\' stato creato');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

const inContesto = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-compressed').map((m) => String(m.content)).join('\n');

const testo = (res) => res.content.map((c) => c.text).join('\n');

test('il nodo vecchio sostituisce i micro con la sintesi, e non perde le foglie', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // 0. Non e' dovuto: nessun nodo. Il tool deve RIFIUTARE, non accorpare il nulla.
    const presto = await tools.get('cwl_old').execute('t', { text: 'SINTESI-PRECOCE' }, undefined, undefined, ctx);
    assert.equal(
      presto.details.ok,
      false,
      'cwl_old ha accorpato quando non c\'era ancora niente da accorpare: la sintesi descriverebbe il vuoto',
    );

    // Cinque foglie, ognuna col suo micro. Con looseLeaves 1 e nodeCapacity 2:
    // la piu' nuova resta sciolta, le altre quattro formano DUE nodi.
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
    }
    const foglie = statoDi(sandbox).spans;
    assert.equal(foglie.length, 5, `attese 5 foglie, ${foglie.length} nello stato`);
    for (let i = 0; i < 5; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro sulla foglia ${i + 1} fallito: ${JSON.stringify(r.details)}`);
    }

    // Un turno per formare i nodi. Il merge e' DOVUTO e va dichiarato.
    const primaDelPozzo = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 12));
    const log = logDi(sandbox).slice(primaDelPozzo);
    assert.match(
      log,
      /OLD NODE due/,
      `con 2 nodi giovani e mergeNodesAt 2 il merge e' dovuto, e nessuna riga lo dice: ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );

    // Prima dell'accorpamento i micro sono TUTTI nel contesto: e' la premessa.
    const prima = inContesto(await hook(hooks, ctx, conversazione(1, 12)));
    for (let i = 1; i <= 5; i++) {
      assert.ok(prima.includes(`MICRO-${i}`), `prima dell'accorpamento manca MICRO-${i}: il fixture non regge`);
    }

    // L'accorpamento: il nodo piu' vecchio (foglie 1 e 2) entra nel pozzo.
    const acc = await tools.get('cwl_old').execute(
      't', { text: 'RIASSUNTONE-1: la sintesi delle prime due storie' }, undefined, undefined, ctx,
    );
    assert.equal(acc.details.ok, true, `cwl_old ha rifiutato quando era dovuto: ${JSON.stringify(acc.details)}`);
    assert.equal(acc.details.nodes, 1, `doveva accorpare UN nodo (il piu' vecchio), ne dichiara ${acc.details.nodes}`);
    assert.equal(acc.details.leaves, 2, `doveva accorpare 2 foglie, ne dichiara ${acc.details.leaves}`);
    const pozzo = acc.details.id;
    assert.match(String(pozzo), /^old-[0-9a-f]{8}$/, `id del pozzo inatteso: ${pozzo}`);

    // Il turno dopo: la sintesi e' nel contesto, i micro accorpati NON ci sono piu'.
    const dopo = inContesto(await hook(hooks, ctx, conversazione(1, 12)));
    assert.ok(dopo.includes('RIASSUNTONE-1'), 'la sintesi non e\' entrata nel contesto: le foglie sono uscite e nessuno sta per loro');
    for (const i of [1, 2]) {
      assert.ok(
        !dopo.includes(`MICRO-${i}`),
        `MICRO-${i} e' ancora nel contesto: le foglie del pozzo vengono ancora iniettate, quindi l'accorpamento non ha liberato niente`,
      );
    }
    // E il nodo GIOVANE no: il presente non entra nel pozzo.
    for (const i of [3, 4, 5]) {
      assert.ok(dopo.includes(`MICRO-${i}`), `MICRO-${i} e' sparito dal contesto: e' finito nel pozzo insieme al passato`);
    }

    // La pagina del pozzo: la sintesi, e la FORMA di cio' che tiene dentro.
    const pagina = await tools.get('cwl_open').execute('t', { id: pozzo }, undefined, undefined, ctx);
    assert.equal(pagina.details.ok, true, `il pozzo non si apre: ${JSON.stringify(pagina.details)}`);
    assert.equal(pagina.details.nodes, 1, `il pozzo dichiara ${pagina.details.nodes} nodi dentro invece di 1`);
    assert.ok(testo(pagina).includes('RIASSUNTONE-1'), 'la pagina del pozzo non riporta la sintesi');
    assert.match(
      testo(pagina),
      /2 leaf\/leaves/,
      `la pagina non dichiara la forma (quante foglie) del nodo che tiene: ${testo(pagina).slice(0, 200)}`,
    );

    // E niente si e' perso: le foglie accorpate si riaprono INTERE.
    const riaperta = await tools.get('cwl_open').execute('t', { id: foglie[0].id }, undefined, undefined, ctx);
    assert.equal(riaperta.details.ok, true, `la foglia accorpata non si riapre: ${JSON.stringify(riaperta.details)}`);
    assert.ok(
      testo(riaperta).includes('CORPO-1'),
      'il corpo della foglia accorpata non torna intero: il risparmio sarebbe un buttare, non un comprimere',
    );
  } finally {
    home.restore();
  }
});
