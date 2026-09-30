/**
 * L'INVARIANTE DELL'INDICE — il primo test del piano (`PIANO-INDICE-RIASSUNTI.md`, sez. 3).
 *
 * Il progetto sta per costruire un indice di riassunti (foglie -> nodo a 30 -> nodo
 * vecchio) che deve restare CONTIGUO e NON SOVRAPPOSTO: nessun buco (niente perso) e
 * nessuna sovrapposizione (niente contato due volte). Prima di scrivere quel codice,
 * l'invariante va provato su cio' che c'e' gia': gli span.
 *
 * DUE BUCHI TROVATI LEGGENDO IL CODICE, e questo test li fissa.
 *
 * 1. `locateSpans` (index.ts 1772-1831) scarta in SILENZIO lo span contenuto in un
 *    altro:
 *
 *      .filter((x, _all, arr) => !arr.some((o) => o !== x && o.from <= x.from && o.to >= x.to))
 *
 *    Il `dead` viene riempito prima, nella `map`, quindi uno span contenuto non
 *    finisce ne' in `resolved` ne' in `dead`: sparisce da tutti i conteggi. E il
 *    log dice `SPANS applied: ${applied.applied}`, cioe' `resolved.length`, quindi
 *    non esiste una riga che dichiari la sparizione. Il suo riassunto resta nello
 *    stato e viene risalvato a ogni turno: peso morto che nessuno conta.
 *
 * 2. Non esiste alcun controllo sulle sovrapposizioni PARZIALI. Il filtro qui sopra
 *    copre solo il contenimento: due span che si intersecano senza contenersi
 *    sopravvivono ENTRAMBI, e la regione condivisa viene contata due volte (anche
 *    da `declareFloor`, che somma `spanTokens`).
 *
 * E' la stessa famiglia dei difetti gia' chiusi in questa sessione (`saved` che era
 * la stima del piano, il risparmio degli span gonfiato): un numero che afferma piu'
 * del lavoro fatto, o una cosa che scompare senza essere dichiarata.
 *
 * Il test costruisce i due span CON GLI STRUMENTI DELL'AGENTE, non a mano: la domanda
 * non e' se `locateSpans` sappia gestire un caso teorico, ma se l'agente possa
 * produrlo. `cwl_compress` (index.ts ~2159-2197) controlla solo che i tre parametri
 * ci siano, che la coppia start+end non esista gia' (allora revoca) e che entrambi
 * gli hash siano in `knownHashes`: nessuna guardia sulle regioni gia' coperte.
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
  const sandbox = makeSandbox({ name: `invariante-${seq++}`, config: config() });
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

/** Lo stato della sessione: il sandbox ne tiene uno solo, quindi non serve l'hash della chiave. */
const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'lo stato della sessione non e\' stato creato');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const compress = (tools, ctx, startHash, endHash, summary) =>
  tools.get('cwl_compress').execute('t', { startHash, endHash, summary }, undefined, undefined, ctx);

/** Il percorso con cui l'agente comprime: lo sceglie l'estensione, e il tool SALVA lo stato. */
const compressRange = (tools, ctx, summary) =>
  tools.get('cwl_compress_range').execute('t', { summary }, undefined, undefined, ctx);

/** Coppie `user`/`assistant`: i turni utente sopravvivono agli span, gli assistant no. */
const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

test('uno span contenuto in un altro non puo\' sparire senza essere dichiarato', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Turno 1: l'estensione prende le misure e sa quale regione comprimere.
    await hook(hooks, ctx, conversazione(1, 6));

    // Span A: il primo, sulla regione che l'estensione ha scelto. Il tool salva lo
    // stato, quindi da qui in poi lo stato e' leggibile su disco.
    const a = await compressRange(tools, ctx, 'SINTESI-A');
    assert.equal(a.details.ok, true, `span A non creato: ${JSON.stringify(a.details)}`);
    await hook(hooks, ctx, conversazione(1, 6));

    // La conversazione CRESCE, cosi' esiste una seconda regione comprimibile: senza
    // di lei non ci sarebbe niente con cui contenere A.
    await hook(hooks, ctx, conversazione(1, 9));
    const b = await compressRange(tools, ctx, 'SINTESI-B');
    assert.equal(b.details.ok, true, `span B non creato: ${JSON.stringify(b.details)}`);

    const dopoB = statoDi(sandbox).spans;
    assert.equal(dopoB.length, 2, `servono DUE span distinti perche' l'invariante abbia un oggetto: ${JSON.stringify(dopoB)}`);
    const [spanA, spanB] = dopoB;

    // Span OUTER: da A.start a B.end. CONTIENE sia A sia B.
    const outer = await compress(tools, ctx, spanA.startHash, spanB.endHash, 'SINTESI-OUTER');
    assert.equal(outer.details.ok, true, `span OUTER non creato: ${JSON.stringify(outer.details)}`);
    assert.equal(statoDi(sandbox).spans.length, 3, 'il terzo span non e\' finito nello stato: la premessa del test non regge');

    // Turno finale: `locateSpans` vede A e B dentro OUTER. Si guarda SOLO cio' che
    // questo turno ha scritto — il log accumula i turni precedenti, e leggere la
    // PRIMA riga invece dell'ultima fa passare il test per il motivo sbagliato.
    // E' successo davvero: la mutazione che applicava E potava gli stessi span
    // (resolved = tutti, dead = i contenuti) passava, perche' la prima riga
    // applicata del log era quella del turno in cui A era nato.
    const primaDelTurno = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 9));
    const log = logDi(sandbox).slice(primaDelTurno);
    const applicati = Number((/SPANS applied: (\d+)/.exec(log) || [0, 0])[1]);
    const potati = Number((/SPANS pruned: (\d+)/.exec(log) || [0, 0])[1]);

    assert.equal(
      applicati + potati,
      3,
      `tre span nello stato, ma il log ne dichiara ${applicati} applicati e ${potati} potati: gli altri sono spariti in silenzio. ` +
        'Il filtro di contenimento di locateSpans li scarta senza registrarli in `dead` (index.ts 1772-1831), ' +
        'quindi non sono ne\' applicati ne\' dichiarati: i loro riassunti restano nello stato a pesare su ogni turno.',
    );
  } finally {
    home.restore();
  }
});
