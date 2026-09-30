/**
 * Il pavimento del contesto dev'essere DICHIARATO, con numeri che si sommano.
 *
 * IL PROBLEMA. `finish` sa gia' rifiutare la richiesta impossibile (controlla
 * `canClose`/`canCompress` prima di chiedere di comprimere), ma non ha mai detto
 * PERCHE' non c'e' niente da consegnare. L'operatore leggeva "il contesto e' sopra
 * il budget" e non poteva distinguere un'estensione rotta da un contesto
 * inviolabile. MISURATO sul vivo: 184.490 token contro un trigger di 68.000 con
 * soli 6.307 token di intervallo comprimibile — il resto era la finestra protetta
 * (gli ultimi 10 turni `user`, il default che l'operatore ha chiesto) e il
 * contenuto che gli span tengono in place.
 *
 * IL CONTRATTO. La riga `CONTEXT ...: Nt in the protected window (last K user
 * turns), Mt inside the spans, Ft freely compressible, Et elsewhere` e' una
 * PARTIZIONE del contesto attivo: i quattro numeri sommano al totale. E'
 * aritmetica, non un racconto, e questo test verifica proprio l'identita' — cosi'
 * una svista nel conto (contare due volte gli span, dimenticare la finestra, non
 * sottrarre `free` da `elsewhere`) diventa un test rosso invece di un numero che
 * sembra plausibile.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Trigger basso (300) e finestra di UN turno: cosi' dopo lo span si resta sopra
 * il trigger (la riga compare) e la finestra protetta contiene davvero qualcosa
 * (altrimenti il test passerebbe con `protectedTokens = 0` per vacuita').
 * Livelli tutti falsi: nessun fallback, il contesto non scende per altre vie.
 */
const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 1,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `floor-declaration-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

test('la riga del pavimento ripartisce il contesto: i quattro numeri sommano al totale', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversazione();
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 dei primi turni' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `lo span non e' stato creato: ${JSON.stringify(comp.details)}`);
    // Lo span si applica; si resta sopra il trigger; senza episodi e senza livelli
    // attivi non c'e' nient'altro da fare, quindi si passa da `finish` sopra budget.
    await hook(hooks, ctx, base);

    const log = logDi(sandbox);
    // L'ULTIMA riga, non la prima: la dichiarazione compare una volta per ogni
    // turno passato da `finish` sopra budget, e nel PRIMO turno (quello che crea
    // l'indirizzo) gli span non esistono ancora, quindi `dentro` sarebbe zero per
    // costruzione e il test non proverebbe niente.
    const righe = [...log.matchAll(/CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window \(last \d+ user turns\), (\d+)t inside the spans, (\d+)t freely compressible, (\d+)t elsewhere/g)];
    assert.ok(righe.length > 0, 'la riga del pavimento non e\' stata scritta: il contesto e\' sopra il trigger e `finish` non ha dichiarato perche\'');
    const [, totale, protetto, dentro, libero, altrove] = righe[righe.length - 1].map(Number);
    assert.equal(
      protetto + dentro + libero + altrove,
      totale,
      `i quattro numeri devono ripartire il contesto: ${protetto} + ${dentro} + ${libero} + ${altrove} != ${totale}`,
    );
    // NON-VACUITA': con finestra di 1 turno e uno span applicato, due dei quattro
    // termini devono essere non nulli, altrimenti l'identita' e' vera per caso.
    assert.ok(protetto > 0, 'la finestra protetta risulta vuota con protectedTurns=1: il test non prova niente');
    assert.ok(dentro > 0, 'il contenuto dentro gli span risulta nullo: il test non prova niente');
  } finally { home.restore(); sandbox.cleanup(); }
});
