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
    // L'ancora di CHIUSURA di uno span e' l'ultimo messaggio ELEGGIBILE sotto il
    // pavimento, e un `assistant` viene RIMOSSO dallo span stesso (dentro uno span
    // sopravvivono solo user/system/developer/custom). Con un turno utente non
    // vuoto appena sotto il pavimento la chiusura sarebbe un `user`,
    // sopravvivrebbe, e il caso di produzione NON si vedrebbe. Un turno utente a
    // corpo VUOTO non e' un indirizzo (sha256("") mapperebbe 1407 messaggi su una
    // voce sola) e viene saltato dal calcolo del range: la chiusura cade allora
    // sull'assistant precedente, che e' esattamente la forma misurata in
    // produzione (`0 of 27 spans located here` con 27 span applicati).
    const conv = conversazione();
    const base = [...conv.slice(0, 10), { role: 'user', content: '' }, ...conv.slice(10)];
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
    const righe = [...log.matchAll(/CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window \(last \d+ user turns\), (\d+)t inside the spans \(counted from (\d+) of (\d+) spans\), (\d+)t freely compressible, (\d+)t elsewhere/g)];
    assert.ok(righe.length > 0, 'la riga del pavimento non e\' stata scritta: il contesto e\' sopra il trigger e `finish` non ha dichiarato perche\'');
    const [, totale, protetto, dentro, contati, tenuti, libero, altrove] = righe[righe.length - 1].map(Number);
    assert.equal(
      protetto + dentro + libero + altrove,
      totale,
      `i quattro numeri devono ripartire il contesto: ${protetto} + ${dentro} + ${libero} + ${altrove} != ${totale}`,
    );
    // La riga DICHIARA da quanti span ha contato. Se il conto viene da una
    // RI-RISOLUZIONE della lista compressa, li' dentro non trova piu' niente
    // (l'ancora di chiusura e' un `assistant`, e lo span l'ha rimossa) e il numero
    // sarebbe 0: e' il difetto che questo test deve uccidere.
    assert.ok(tenuti > 0, 'il test non ha creato nessuno span: non prova niente');
    assert.equal(
      contati,
      tenuti,
      `il conto deve venire da TUTTI i ${tenuti} span tenuti, non da ${contati}: se viene da una ri-risoluzione della lista compressa, le ancore di chiusura non ci sono piu'`,
    );
    // NON-VACUITA': con finestra di 1 turno e uno span applicato, due dei quattro
    // termini devono essere non nulli, altrimenti l'identita' e' vera per caso.
    assert.ok(protetto > 0, 'la finestra protetta risulta vuota con protectedTurns=1: il test non prova niente');
    assert.ok(dentro > 0, 'il contenuto dentro gli span risulta nullo: il test non prova niente');
    // Lo spacchettamento per ruolo. Le tre parti devono sommare al totale della
    // riga SOPRA (due righe, un numero: non si puo' raccontare), e nel fixture
    // dentro lo span sopravvivono solo i turni `user` e il riassunto iniettato:
    // NIENTE system/developer/custom. Se la classificazione sbaglia, o `utente`
    // va a zero, o `altro` smette di essere zero.
    const contenuto = [...log.matchAll(/SPANS content: (\d+)t inside the spans = (\d+)t of summaries \+ (\d+)t of user turns \+ (\d+)t of other roles/g)];
    assert.ok(contenuto.length > 0, 'la riga dello spacchettamento non e\' stata scritta con uno span applicato: non si puo\' sapere cosa gli span tengono');
    const [, dentro2, riassunti, utente, altro] = contenuto[contenuto.length - 1].map(Number);
    assert.equal(dentro2, dentro, `lo spacchettamento deve riguardare lo stesso totale della riga CONTEXT: ${dentro2} != ${dentro}`);
    assert.equal(riassunti + utente + altro, dentro, `le tre parti devono sommare al totale: ${riassunti} + ${utente} + ${altro} != ${dentro}`);
    assert.ok(utente > 0, 'dentro lo span devono sopravvivere i turni utente: se il conto non li vede, la classificazione e\' rotta');
    assert.ok(riassunti > 0, 'dentro lo span c\'e\' il riassunto iniettato: se il conto non lo vede, la classificazione e\' rotta');
    assert.equal(altro, 0, `nel fixture non c'e' nessun messaggio system/developer/custom dentro lo span, ma ne risultano ${altro}t: la classificazione sta contando la cosa sbagliata`);
  } finally { home.restore(); sandbox.cleanup(); }
});
