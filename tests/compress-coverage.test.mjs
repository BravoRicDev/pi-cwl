/**
 * UNA REGIONE GIA' DENTRO UNA FOGLIA NON SI RICOMPRIME.
 *
 * `compressibleRange` salta ogni indice che sta dentro uno span risolto, quindi la strada
 * OFFERTA (`cwl_compress_range`) non puo' descrivere due volte gli stessi messaggi. Il tool
 * ESPLICITO (`cwl_compress`) no: validava i due hash contro `knownHashes` e spingeva la foglia,
 * senza sapere niente della copertura. Un agente che sceglie gli hash a mano poteva quindi
 * creare una foglia che si sovrappone a una esistente — la regione descritta due volte, e le
 * due descrizioni libere di divergere col tempo.
 *
 * Il tool non ha accesso alla lista dei messaggi (lo dice il suo stesso commento), ma il
 * `sessionManager` del contesto espone `buildContextEntries()`, che e' cio' da cui Pi costruisce
 * la lista del contesto: da li' si ricostruisce lo STESSO ORDINE di indirizzi, e per un
 * confronto fra intervalli l'ordine e' tutto cio' che serve.
 *
 * Due casi, e servono entrambi:
 *  (1) un intervallo dentro una foglia viene RIFIUTATO, con la ragione nel log;
 *  (2) la STESSA chiamata, quando gli estremi non sono collocabili sulla lista letta, PASSA e
 *      dichiara di non aver controllato. Un controllo che rifiuta sempre sarebbe indistinguibile
 *      da un controllo onesto senza questo secondo caso.
 *
 * DICHIARATO — cosa questo test NON uccide: la mutazione che fa rispondere a `covers` sempre
 * "coperto" resta VIVA, perche' il terzo caso che servirebbe — un candidato COLLOCABILE e NON
 * coperto — non esiste in questo fixture: con `protectedTurns: 0` la foglia copre tutta la lista,
 * quindi ogni hash che il tool accetta cade dentro di lei. Il caso (2) non la uccide perche' esce
 * prima, sul ramo `unknown`. Serve un fixture con una coda scoperta (per esempio `protectedTurns`
 * piu' alto, e un hash preso da quella coda). Meglio dirlo che lasciar credere che sia coperto.
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
  const sandbox = makeSandbox({ name: `copertura-${seq++}`, config: config() });
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
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('un intervallo dentro una foglia viene rifiutato, e la stessa cosa senza copertura passa', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversazione(1, 7);
    await hook(hooks, ctx, base); // il hook calcola l'indirizzo offerto
    ctx.__mostra(base); // questa e' la "sessione" che i tool vedranno

    const primo = await tools
      .get('cwl_compress_range')
      .execute('t', { summary: 'CORPO-1 ' + 'x'.repeat(300) }, undefined, undefined, ctx);
    assert.equal(primo.details.ok, true, `la foglia non e' nata: ${JSON.stringify(primo.details)}`);

    const foglia = statoDi(sandbox).spans[0];
    assert.ok(foglia && foglia.startHash, 'nessuna foglia con indirizzo nello stato: il test non prova niente');
    // `cwl_compress` valida i due hash contro `knownHashes`, che contiene l'hash del TESTO di
    // ogni messaggio (non l'indirizzo `id|testo` che portano le foglie: sono due namespace
    // diversi, e `locateSpans` li risolve entrambi via `addressMaps`). Il test prende quindi un
    // hash dal set che il tool accetta — il piu' vecchio — e lo usa come estremo.
    const stato = statoDi(sandbox);
    const h = stato.knownHashes[0];
    assert.ok(typeof h === 'string' && h.length > 0, 'nessun hash persistito: il tool rifiuterebbe per unknown-hash');

    // (1) Un intervallo che cade DENTRO la foglia: rifiutato, con la ragione nel log.
    const primaDentro = logDi(sandbox).length;
    const dentro = await tools
      .get('cwl_compress')
      .execute('t', { startHash: h, endHash: h, summary: 'CORPO-2' }, undefined, undefined, ctx);
    assert.equal(
      dentro.details.error,
      'covered-range',
      `atteso il rifiuto della copertura, arrivato ${JSON.stringify(dentro.details)}`,
    );
    assert.match(
      logDi(sandbox).slice(primaDentro),
      /COMPRESS refused [0-9a-f]+\.\.[0-9a-f]+: inside a live leaf \(\d+ span\(s\) resolved, \d+ message\(s\) read\)/,
      'il rifiuto non dice perche\' e non dichiara su quanti messaggi ha guardato',
    );
    assert.equal(statoDi(sandbox).spans.length, 1, 'la foglia rifiutata e\' finita nello stato lo stesso');

    // (2) NON-VACUITA': la stessa chiamata con gli estremi NON collocabili sulla lista letta
    //     passa, e il log deve DICHIARARE che non ha controllato (mai accettare in silenzio).
    //     La finta sessione mostra solo l'ULTIMO messaggio: il piu' vecchio hash del set non e'
    //     piu' collocabile su quella lista.
    ctx.__mostra(base.slice(-1));
    const primaFuori = logDi(sandbox).length;
    const fuori = await tools
      .get('cwl_compress')
      .execute('t', { startHash: h, endHash: h, summary: 'CORPO-3' }, undefined, undefined, ctx);
    assert.equal(
      fuori.details.ok,
      true,
      `atteso il passaggio dichiarato, arrivato ${JSON.stringify(fuori.details)}: un controllo che non sa collocare ` +
        'gli estremi non deve bocciare una richiesta legittima',
    );
    assert.match(
      logDi(sandbox).slice(primaFuori),
      /COMPRESS coverage: endpoints of .* are not placeable on the \d+ message\(s\) read — let through, NOT checked/,
      'il passaggio non e\' stato dichiarato nel log: un controllo che non gira in silenzio e\' la stessa cosa di un controllo assente',
    );
  } finally {
    home.restore();
  }
});
