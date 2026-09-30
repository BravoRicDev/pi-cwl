/**
 * Il risparmio che uno span DICHIARA deve essere quello che il contesto ha perso.
 *
 * IL DIFETTO. In `applySpans` il guadagno di uno span era calcolato su tutto il
 * range:
 *
 *   for (let i = from; i <= to; i++) { replaced.add(i); original += estimateMessageTokens(messages[i]); }
 *   const gain = Math.max(0, original - estimateTokens(sp.summary));
 *
 * ma il range NON viene rimosso per intero: piu' sotto, quando la lista viene
 * ricostruita, i messaggi `user`, `system`, `developer` e `custom` che cadono
 * dentro lo span vengono TENUTI (`if (role === 'user' || ...) { kept.push(m); }`).
 * Quei token restano nel contesto e continuano a costare, ma il guadagno li
 * contava come risparmiati. MISURATO in un sandbox con una conversazione di 12
 * messaggi (6 `user` + 6 `assistant`, ~756 token): il log dichiarava
 * `SPANS applied: 1 (new 1), saved 749t` mentre i sei messaggi `user` erano
 * ancora presenti nella lista restituita, quindi il contesto era sceso di ~338t.
 * Lo span dichiarava piu' del doppio di quanto aveva liberato.
 *
 * E' la stessa famiglia del difetto chiuso nella contabilita' dell'evacuazione
 * (`saved` era la stima del piano invece di `B-A`): un numero che afferma piu' del
 * lavoro fatto. La conseguenza non e' cosmetica: `cwl_status` somma quei
 * risparmi, quindi l'estensione crede di aver liberato spazio che ha ancora, e
 * smette di cercare leve proprio quando il contesto e' ancora sopra il trigger
 * (dal vivo: ~142.000 token misurati contro un trigger di 68.000).
 *
 * Lo stesso numero sbagliato finiva nel messaggio iniettato al posto dei
 * messaggi compressi, che dichiara `(~N token risparmiati)` passando la
 * dimensione del RANGE.
 *
 * Il rimedio e' quello dell'evacuazione: contare cio' che si e' DAVVERO rimosso,
 * con un'unica definizione dei ruoli che sopravvivono usata sia dal conto sia
 * dalla ricostruzione della lista, cosi' i due non possono piu' divergere.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Budget basso di proposito: il trigger (300) deve restare sotto il contesto
 * anche DOPO lo span, perche' e' la riga che l'estensione scrive in quel caso
 * (`SPANS applied (Nt) still above trigger Mt`) a darci la misura del dopo.
 */
const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `span-accounting-${seq++}`, config: config() });
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

/** 6 `user` + 6 `assistant`: meta' del range sopravvive allo span (i turni utente). */
const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

test('il risparmio dichiarato da uno span e\' quello che il contesto ha davvero perso', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversazione();
    // Turno 1: l'estensione prende le misure e memorizza l'indirizzo da comprimere.
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 dei primi turni' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `lo span non e' stato creato (senza span il test non prova niente): ${JSON.stringify(comp.details)}`);

    // Turno 2: lo span si applica per la prima volta.
    const out = await hook(hooks, ctx, base);
    const log = logDi(sandbox);

    const prima = /RANGE \S+ \(~\d+t\) \| 12 msgs, (\d+)t vs trigger/.exec(log);
    const dichiarato = /SPANS applied: \d+ \(new \d+\), saved (\d+)t/.exec(log);
    const dopo = /SPANS applied \((\d+)t\) still above trigger/.exec(log);
    assert.ok(
      prima && dichiarato && dopo,
      'servono tutte e tre le righe (misura prima, risparmio dichiarato, misura dopo): ' +
      `prima=${!!prima} dichiarato=${!!dichiarato} dopo=${!!dopo}`,
    );

    // NON-VACUITA': i turni `user` dentro lo span sopravvivono davvero, quindi il
    // dichiarato ha modo di esagerare. Se un giorno lo span li rimuovesse, questo
    // test va ripensato invece di restare verde per caso.
    assert.ok(JSON.stringify(out).includes('turno 1 contenuto'),
      'i turni `user` dentro lo span risultano rimossi: la premessa del test non vale piu\'');

    assert.equal(
      Number(dichiarato[1]),
      Number(prima[1]) - Number(dopo[1]),
      `lo span dichiara di aver risparmiato ${dichiarato[1]}t ma il contesto e' sceso da ${prima[1]}t a ${dopo[1]}t, ` +
      `cioe' ${Number(prima[1]) - Number(dopo[1])}t: i turni utente dentro lo span restano nel contesto e continuano a costare`,
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('il messaggio iniettato non dichiara piu\' token di quanti il contesto ne abbia persi', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversazione();
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 dei primi turni' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `lo span non e' stato creato: ${JSON.stringify(comp.details)}`);

    const out = await hook(hooks, ctx, base);
    const log = logDi(sandbox);
    const prima = /RANGE \S+ \(~\d+t\) \| 12 msgs, (\d+)t vs trigger/.exec(log);
    const dopo = /SPANS applied \((\d+)t\) still above trigger/.exec(log);
    assert.ok(prima && dopo, 'misure mancanti: il test non prova niente');
    const persi = Number(prima[1]) - Number(dopo[1]);

    const iniettato = out.find((m) => m?.customType === 'cwl-compressed');
    assert.ok(iniettato, 'il riepilogo compresso non e\' stato iniettato');
    const claim = /~(\d+) token risparmiati/.exec(JSON.stringify(iniettato));
    assert.ok(claim, 'il messaggio iniettato non dichiara i token risparmiati');
    assert.ok(
      Number(claim[1]) <= persi,
      `il messaggio iniettato dichiara ~${claim[1]} token risparmiati ma il contesto ne ha persi ${persi}: ` +
      'il numero era la dimensione del RANGE, non il risparmio',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});
