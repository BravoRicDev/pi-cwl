/**
 * `cwl_open`: una foglia si riapre INTERA, e se non c'e' lo dice.
 *
 * IL PROBLEMA CHE RISOLVE. Un riassunto vive SOLO nello stato: il transcript e'
 * append-only e tiene i MESSAGGI ORIGINALI, non i riassunti. Finche' l'unica via
 * per rileggere cio' che una compressione ha messo da parte era ricomprare i
 * messaggi dal transcript, il lavoro dell'agente (il riassunto) era scrivibile e
 * non rileggibile. `cwl_open` chiude quel cerchio.
 *
 * DUE REGOLE, e sono entrambe richieste dell'operatore:
 *  1. il corpo torna INTERO, senza troncamento. Non e' un dettaglio di comodo:
 *     i suoi modelli hanno finestre da 1M di token e un agente che vuole vedere
 *     qualcosa deve VEDERLO. L'unico obbligo del tool e' dichiarare la dimensione
 *     PRIMA di consegnarla, non tagliarla;
 *  2. un id che non esiste viene DICHIARATO. Il silenzio e' il modo in cui i bug
 *     si nascondono, e qui sarebbe il peggiore: l'agente crederebbe di avere in
 *     mano il contenuto e non ce l'ha.
 *
 * Il secondo caso e' quello che rende onesto il primo: senza, un `cwl_open` che
 * risponde "ecco il corpo" a un id inesistente passerebbe il test.
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
  const sandbox = makeSandbox({ name: `open-${seq++}`, config: config() });
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

const testo = (res) => res.content.map((c) => c.text).join('\n');

test('cwl_open restituisce il corpo INTERO e dichiara un id che non esiste', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await hook(hooks, ctx, conversazione());

    // Un corpo molto piu' lungo di qualunque anteprima: se il tool tronca, si vede.
    const corpo = 'RIASSUNTO-INTEGRALE ' + 'x'.repeat(8000) + ' FINE-DEL-CORPO';
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: corpo }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `lo span non e' stato creato: ${JSON.stringify(comp.details)}`);

    const sp = statoDi(sandbox).spans[0];
    assert.ok(sp && typeof sp.id === 'string' && sp.id.length > 0, `lo span non ha un id stabile: ${JSON.stringify(sp)}`);

    const res = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    const out = testo(res);
    assert.equal(res.details.ok, true, `cwl_open ha rifiutato un id che esiste: ${JSON.stringify(res.details)}`);

    // 1. Il corpo torna INTERO: non "un estratto", non "i primi N caratteri".
    assert.ok(
      out.includes(corpo),
      `il corpo non e' tornato intero: il tool ha risposto con ${out.length} caratteri, il corpo ne ha ${corpo.length}. ` +
        'Un riassunto troncato e\' un riassunto perso: il testo originale sta nel transcript, il riassunto no.',
    );

    // 2. La dimensione e' dichiarata PRIMA di consegnarlo (l'unico obbligo).
    assert.match(out, /~\d+ token/, `il tool non dichiara la dimensione del corpo: ${out.slice(0, 120)}`);

    // 3. Un id inesistente viene DICHIARATO, non taciuto.
    const nessuno = await tools.get('cwl_open').execute('t', { id: 'sp-00000000' }, undefined, undefined, ctx);
    assert.equal(
      nessuno.details.ok,
      false,
      'un id inesistente e\' stato accettato: l\'agente crederebbe di avere in mano un contenuto che non esiste',
    );
    assert.match(
      testo(nessuno),
      /sp-00000000/,
      `la dichiarazione non nomina l'id chiesto: ${testo(nessuno).slice(0, 160)}`,
    );

    // 4. La foglia deve essere RAGGIUNGIBILE: l'id che l'agente vede nel contesto
    //    (l'avviso iniettato al posto dei messaggi compressi) deve essere quello che
    //    `cwl_open` accetta. Un id che nessuno vede e' un tool che nessuno puo'
    //    usare, e nessun test se ne accorgerebbe finche' non ci si prova davvero.
    const conAvviso = await hook(hooks, ctx, conversazione());
    const iniettato = conAvviso.find((m) => m && m.customType === 'cwl-compressed');
    assert.ok(iniettato, 'nessun messaggio di compressione nel contesto: la foglia non e\' nemmeno nominata');
    const visto = /(sp-[0-9a-f]{8})/.exec(String(iniettato.content ?? ''));
    assert.ok(
      visto,
      `l'avviso nel contesto non dice quale id aprire: ${String(iniettato.content).slice(0, 140)}`,
    );
    const riaperto = await tools.get('cwl_open').execute('t', { id: visto[1] }, undefined, undefined, ctx);
    assert.equal(
      riaperto.details.ok,
      true,
      `l'id visto nel contesto (${visto[1]}) non e' apribile: ${JSON.stringify(riaperto.details)}`,
    );
  } finally {
    home.restore();
  }
});
