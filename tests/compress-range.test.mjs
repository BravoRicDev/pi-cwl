/**
 * `cwl_compress_range`: l'estensione calcola l'indirizzo, il modello scrive il testo.
 *
 * Perche' esiste. Il budget gate chiedeva di compattare con
 * `cwl_compress(startHash=..., endHash=...)`. In una sessione VERA quella richiesta
 * non era eseguibile: gli hash vivono nello stato, sono opachi e l'agente non ha
 * modo di sapere quale hash corrisponda a quale messaggio. Risultato misurato:
 * gate armato al turno 2, 3 tentativi, zero compattazioni, contesto fermo a 370k
 * contro una soglia di 68k.
 *
 * La divisione che funziona: l'estensione sceglie gli indirizzi (li ha), il
 * modello scrive il riassunto (l'unica parte che solo il modello puo' fare).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const config = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 2,
  gate: false,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(cfg) {
  const sandbox = makeSandbox({ name: `range-${seq++}`, config: cfg });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Sei turni user+assistant: gli ultimi 2 restano nella finestra di sicurezza. */
const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

const call = (tools, ctx, summary) =>
  tools.get('cwl_compress_range').execute('t', { summary }, undefined, undefined, ctx);

test('cwl_compress_range comprime l\'intervallo piu\' vecchio, senza hash', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, 'sintesi dei primi turni: obiettivo, decisioni, path');
    assert.equal(out.details.ok, true, `rifiutato: ${JSON.stringify(out.details)}`);
    assert.ok(out.details.tokens > 0, 'l\'intervallo compresso deve avere token');
    // L'indirizzo e' stato consumato: un secondo colpo non puo' ricomprimere lo stesso.
    const again = await call(tools, ctx, 'secondo tentativo');
    assert.equal(again.details.ok, false);
    assert.equal(again.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la finestra di sicurezza non viene toccata: l\'intervallo si ferma prima', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 2 }));
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'sintesi');
    assert.equal(out.details.ok, true);

    // Verifica diretta: gli hash dell'intervallo NON devono includere i messaggi
    // degli ultimi 2 turni (gli ultimi 4 messaggi della lista).
    const { createHash } = await import('node:crypto');
    const hashOf = (s) => createHash('sha256').update(s).digest('hex').slice(0, 12);
    const protetti = messages.slice(-4).map((m) => hashOf(m.content));
    assert.ok(!protetti.includes(out.details.startHash ?? ''), 'start dentro la finestra');
    assert.ok(!protetti.includes(out.details.endHash ?? ''), 'end dentro la finestra');
    // E deve invece includere il primo messaggio (fuori dalla finestra).
    assert.equal(out.content[0].text.includes('Compresso'), true);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('senza niente da comprimere rifiuta invece di inventare un intervallo', async () => {
  // Due messaggi soli: prima della finestra non c'e' abbastanza materiale per
  // formare un intervallo (serve una coppia), quindi non c'e' niente da prendere.
  // Il caso "finestra tanto larga da coprire tutto" NON e' piu' un rifiuto: la
  // parte piu' vecchia viene liberata apposta, altrimenti l'estensione non
  // potrebbe mai chiudere il contesto (misurato: 469k token, soglia 68k, nessuna
  // compattazione possibile).
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 99 }));
  try {
    const minimale = [
      { role: 'user', content: 'domanda ' + 'U'.repeat(200) },
      { role: 'assistant', content: 'risposta ' + 'A'.repeat(200) },
    ];
    await hooks.get('context')({ messages: minimale }, ctx);
    const out = await call(tools, ctx, 'sintesi');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('un riassunto vuoto viene rifiutato: e\' l\'unica parte che deve scrivere il modello', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, '   ');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'missing-summary');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('un intervallo gia\' compresso non viene riproposto', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const first = await call(tools, ctx, 'prima sintesi');
    assert.equal(first.details.ok, true);
    // Il hook successivo ricalcola l'intervallo SU QUELLO CHE RESTA.
    await hooks.get('context')({ messages }, ctx);
    const status = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
    const testo = status.content.map((c) => c.text).join('\n');
    // O non resta niente, o resta un intervallo diverso dal primo.
    const hasRange = /Intervallo comprimibile/.test(testo);
    if (hasRange) {
      assert.ok(!testo.includes(`${first.details.startHash}..${first.details.endHash}`),
        'l\'intervallo gia\' compresso viene riproposto');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('il risparmio si conta UNA volta: riapplicare lo span non gonfia il totale', async () => {
  // Pi ricostruisce la lista dal transcript a ogni turno, quindi rimanda gli
  // ORIGINALI: lo span dev'essere riapplicato, ed e' giusto che sia cosi'.
  // Riapplicare non e' risparmiare di nuovo — il contesto resta quello
  // compresso, non si comprime due volte. Contare il risparmio a ogni
  // riapplicazione produce un numero che cresce da solo, turno dopo turno, e
  // quel numero e' l'unica prova che l'operatore ha che la cosa funzioni.
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  const risparmiati = async () => {
    const s = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
    const testo = s.content.map((c) => c.text).join('\n');
    const m = /token risparmiati:\s*([\d.]+)/.exec(testo);
    return m ? Number(m[1].replace(/\./g, '')) : -1;
  };
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'sintesi dei primi turni: obiettivo, decisioni, path');
    assert.equal(out.details.ok, true);
    // Il conteggio avviene quando lo span viene APPLICATO, cioe' al turno
    // successivo: il tool scrive l'indirizzo, l'hook lo applica.
    await hooks.get('context')({ messages }, ctx);
    const dopoLaCompressione = await risparmiati();
    assert.ok(dopoLaCompressione > 0, 'la compressione deve aver risparmiato qualcosa');

    // Tre turni in cui si rimanda esattamente la stessa lista: nessuna nuova
    // compressione, nessuna nuova eviction. Il totale non deve muoversi.
    for (let i = 0; i < 3; i++) await hooks.get('context')({ messages }, ctx);
    const dopoTreTurni = await risparmiati();
    assert.equal(dopoTreTurni, dopoLaCompressione,
      `il risparmio e' cresciuto senza nuove compressioni: ${dopoLaCompressione} -> ${dopoTreTurni}`);
  } finally { home.restore(); sandbox.cleanup(); }
});

/** Coppie toolCall/toolResult rotte: sono esattamente cio' che il provider rifiuta. */
const orfani = (messages) => {
  const chiamate = new Set();
  const risultati = new Set();
  for (const m of messages) {
    if (Array.isArray(m.content)) {
      for (const b of m.content) if (b && b.type === 'toolCall' && b.id) chiamate.add(b.id);
    }
    if (typeof m.toolCallId === 'string') risultati.add(m.toolCallId);
  }
  return {
    senzaRisultato: [...chiamate].filter((id) => !risultati.has(id)),
    senzaChiamata: [...risultati].filter((id) => !chiamate.has(id)),
  };
};

/**
 * Regressione: la compressione non deve spezzare una coppia toolCall/toolResult.
 *
 * Misurato su una sessione VERA. L'intervallo compresso finiva su un assistant
 * che portava una toolCall (indice 1296) mentre il suo toolResult, 6396 char,
 * restava fuori (indice 1297). Nel contesto sopravviveva un `tool_result` senza
 * il suo `tool_use`, il provider rispondeva `400 status code (no body)` e la
 * sessione si bloccava: nessun messaggio di errore utile, nessun modo di
 * riprendere se non a mano.
 *
 * Il percorso di eviction deterministica conosceva GIA' questo invariante e lo
 * rispettava (H1, `droppedToolCallIds`: "the assistant message that carries the
 * matching toolCall must not keep it, or the conversation has a dangling...").
 * Il percorso degli span non aveva alcuna guardia.
 *
 * Perche' il caso si presenta solo ORA: nel caso normale `protectedFromIndex`
 * restituisce `indice_user + 1`, quindi il range finisce sempre su un messaggio
 * user. Solo nel caso DEGENERATO (meno turni utente della finestra) il pavimento
 * e' una quota arbitraria della lista, e allora puo' cadere subito dopo un
 * assistant. E' la stessa condizione del fix "la finestra di sicurezza copriva
 * TUTTA la lista": 469k token contro 68k di soglia.
 */
test('la compressione non lascia un toolResult orfano', async () => {
  // protectedTurns 4 con 2 soli turni utente: caso degenerato, il pavimento
  // cade a meta' lista — esattamente dove e' caduto nella sessione vera.
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 4 }));
  try {
    const testo = (t) => `${t} ` + 'X'.repeat(240);
    const scambio = (n) => ([
      { role: 'assistant', content: [{ type: 'text', text: testo(`penso ${n}`) }, { type: 'toolCall', id: `tc${n}`, name: 'bash', arguments: { command: 'ls' } }] },
      { role: 'toolResult', toolCallId: `tc${n}`, content: [{ type: 'text', text: testo(`output ${n}`) }] },
    ]);
    const messages = [
      { role: 'user', content: testo('turno 1') },
      ...scambio(1),
      ...scambio(2),
      ...scambio(3),
      { role: 'user', content: testo('turno 2') },
    ];

    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'sintesi dei turni con strumenti');
    assert.equal(out.details.ok, true, `rifiutato: ${JSON.stringify(out.details)}`);

    // L'hook e' l'unico punto in cui la lista viene riscritta: e' cio' che va al provider.
    const res = await hooks.get('context')({ messages }, ctx);
    const kept = res.messages;
    const { senzaRisultato, senzaChiamata } = orfani(kept);
    assert.deepEqual(senzaChiamata, [],
      `tool_result senza il suo tool_use: il provider risponde 400. Orfani: ${senzaChiamata.join(', ')}`);
    assert.deepEqual(senzaRisultato, [],
      `tool_use senza il suo risultato: il provider risponde 400. Orfani: ${senzaRisultato.join(', ')}`);
    // La finestra di sicurezza resta intoccabile: estendere l'intervallo ai
    // toolResult non deve diventare "mangia tutto fino in fondo".
    assert.ok(kept.some((m) => m.role === 'user' && String(m.content).includes('turno 2')),
      'il turno utente protetto deve sopravvivere');
  } finally { home.restore(); sandbox.cleanup(); }
});
