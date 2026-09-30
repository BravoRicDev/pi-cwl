/**
 * IL TETTO DI UN'ETICHETTA, E L'ECCESSO DETTO FUORI.
 *
 * MISURATO su 42 etichette reali: ~722t l'una (~1.900 caratteri) contro le ~300t che il design
 * dava per scontate. La cima dell'indice e' fatta di etichette, quindi costava 30k invece di
 * ~12k. Il prompt diceva "~200 parole" e non vincolava niente: 42 etichette sono uscite a ~300
 * parole l'una. Un tetto in CARATTERI e' una cosa che un modello puo' contare mentre scrive.
 *
 * L'eccesso NON viene rifiutato: rifiutarlo bloccherebbe il lavoro, e un'etichetta troppo lunga
 * resta un'etichetta. Viene DICHIARATO, nel risultato del tool e nel log. Cio' che non e'
 * accettabile e' il silenzio.
 *
 * Due casi, e servono entrambi: sopra il tetto la dichiarazione DEVE esserci; sotto il tetto NON
 * deve esserci, altrimenti "dichiara l'eccesso" sarebbe indistinguibile da "dichiara sempre".
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;
const TETTO = 1400;

const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `tetto-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

/** Guida l'hook (che calcola l'indirizzo) e poi comprime con l'etichetta richiesta. */
async function comprimi(sandbox, hooks, ctx, tools, etichetta) {
  const base = conversazione(1, 7);
  await hooks.get('context')({ messages: base }, ctx);
  const prima = logDi(sandbox).length;
  const res = await tools
    .get('cwl_compress_range')
    .execute('t', { summary: 'CORPO-1 ' + 'x'.repeat(300), micro: etichetta }, undefined, undefined, ctx);
  const testo = res.content.map((c) => c.text).join('\n');
  return { res, testo, dopo: logDi(sandbox).slice(prima) };
}

test("un'etichetta oltre il tetto viene REGISTRATA e l'eccesso DICHIARATO nel risultato e nel log", async () => {
  const { sandbox, home, hooks, ctx, tools } = await boot();
  try {
    const { res, testo, dopo } = await comprimi(sandbox, hooks, ctx, tools, 'E'.repeat(TETTO + 100));
    assert.equal(res.details.ok, true, `la compressione non e' passata: ${JSON.stringify(res.details)}`);
    assert.match(
      testo,
      new RegExp(String(TETTO)),
      "il risultato non nomina il tetto: l'agente non sa che ha sfondato la misura, e la prossima etichetta sara' lunga uguale",
    );
    assert.match(
      dopo,
      /MICRO over ceiling: \S+ is \d+ characters \(~\d+t\) against a ceiling of \d+ \(~\d+t\)/,
      'il log non ha misurato l\'eccesso: nessuna riga, nessun numero, nessun modo di accorgersene',
    );
  } finally {
    home.restore();
  }
});

test("un'etichetta sotto il tetto non produce nessuna dichiarazione (la misura non e' un rito)", async () => {
  const { sandbox, home, hooks, ctx, tools } = await boot();
  try {
    const { res, testo, dopo } = await comprimi(sandbox, hooks, ctx, tools, 'E'.repeat(TETTO - 100));
    assert.equal(res.details.ok, true, `la compressione non e' passata: ${JSON.stringify(res.details)}`);
    assert.doesNotMatch(testo, new RegExp(String(TETTO)), 'il risultato nomina il tetto anche quando non e\' stato sfondato');
    assert.doesNotMatch(dopo, /MICRO over ceiling/, 'il log dichiara un eccesso che non c\'e\' stato');
  } finally {
    home.restore();
  }
});
