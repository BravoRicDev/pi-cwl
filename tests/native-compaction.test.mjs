/**
 * Livello A: l'incontro con la compattazione nativa di Pi.
 *
 * Perche' esiste. La compattazione nativa di Pi taglia un PREFISSO: tiene
 * `firstKeptEntryId` e tutto cio' che viene dopo, e antepone il proprio
 * riepilogo. Due conseguenze, entrambe verificate qui.
 *
 * 1. Un episodio ancora aperto quando il taglio avviene perde la sua ancora di
 *    APERTURA e conserva quella di chiusura. La sua parte sopravvissuta e'
 *    tutto cio' che la lista tiene ancora fino a quella chiusura, quindi il
 *    range va DEDOTTO ([0, end]) invece che letto.
 *
 * 2. Quel range parte dall'indice 0, dove vive il riepilogo della compattazione
 *    nativa: un messaggio con role 'compactionSummary'
 *    (pi/dist/core/messages.js, createCompactionSummaryMessage) che NON e' un
 *    turno utente e che nessuno proteggeva. Evacuarlo distrugge la sola copia
 *    della storia che la compattazione ha sostituito: il transcript la tiene, il
 *    provider no.
 *
 * La direzione opposta (chiusura persa, apertura viva -> [start, len-1]) e'
 * RIFIUTATA, e c'e' un test che la tiene rifiutata: un taglio di prefisso non
 * puo' portare via la chiusura lasciando l'apertura, quindi quella disposizione
 * non ha spiegazione, e inventare un range per una disposizione inspiegata e'
 * esattamente il modo in cui questa estensione evacuerebbe cio' che non sa
 * contare.
 *
 * MISURATO prima di scrivere il ramo (189 transcript, 202 compattazioni native,
 * 7 episodi chiusi): 5 episodi stavano interamente PRIMA del taglio — le loro
 * DUE ancore erano sparite, e nessuna deduzione puo' salvarli — 2 interamente
 * dopo, 0 a cavallo. Il ramo quindi non ripara niente che sia stato osservato:
 * e' un'assicurazione per la disposizione che il taglio rende possibile. Ecco
 * perche' il test pretende anche che il LOG dica che e' successo.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Solo l'evacuazione piena: isola la deduzione dai tre livelli di strip. Con i
 * livelli di strip spenti l'escalation del piano non ha altra scelta che
 * arrivare a `removed`, che e' il livello dove il range conta davvero.
 */
const soloRimozione = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(config) {
  const sandbox = makeSandbox({ name: `native-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => {
  const p = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
};

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const testo = (m) => JSON.stringify(m ?? {});
const contiene = (out, marker) => out.some((m) => testo(m).includes(marker));
/** Riempitivo abbastanza grande da rendere necessario il livello `removed`. */
const assistant = (marker) => ({
  role: 'assistant',
  content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }],
});

const RIEPILOGO = 'RIEPILOGO NATO DAL TAGLIO ' + 'S'.repeat(4000);
const riepilogo = () => ({
  role: 'compactionSummary',
  summary: RIEPILOGO,
  tokensBefore: 90000,
  timestamp: 1,
});

/** Episodio nato PRIMA del taglio: l'apertura non e' piu' nella lista. */
async function apriEChiudi(tools, ctx, nome, tipo = 'expl') {
  await tools.get('delimiter').execute('call-s', { action: 'start', name: nome, type: tipo }, undefined, undefined, ctx);
  await tools.get('delimiter').execute('call-e', { action: 'end', name: nome, description: 'imparato' }, undefined, undefined, ctx);
}

test('un episodio che ha perso l\'apertura viene localizzato per deduzione e evacuato', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    await apriEChiudi(tools, ctx, 'attraversa');
    const messages = [
      riepilogo(),
      assistant('DENTRO-1'),
      assistant('DENTRO-2'),
      // L'ancora di chiusura, viva: e' l'unico estremo rimasto. Vive all'indice 3,
      // quindi il range dedotto e' [0, 3].
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(!contiene(out, 'DENTRO-1') && !contiene(out, 'DENTRO-2'),
      'il range non e\' stato dedotto: l\'episodio e\' rimasto invisibile all\'evacuazione');
    // Il log deve dirlo. Un range dedotto invece che letto e' una cosa che
    // l'operatore deve poter vedere, altrimenti la deduzione e' un'assunzione.
    assert.match(logDi(sandbox), /EPISODES deduced: 1/, 'il log non ha detto che un range e\' stato dedotto');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('il riepilogo della compattazione nativa sopravvive a quel range', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    await apriEChiudi(tools, ctx, 'attraversa');
    const messages = [
      riepilogo(),
      assistant('DENTRO-1'),
      assistant('DENTRO-2'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    // Il range dedotto e' stato applicato (altrimenti questo test non proverebbe
    // niente sul riepilogo: sarebbe vivo solo perche' nessuno l'ha toccato).
    assert.ok(!contiene(out, 'DENTRO-1'), 'il range non e\' stato applicato: il test non prova niente');
    assert.ok(contiene(out, RIEPILOGO),
      'il riepilogo della compattazione nativa e\' stato evacuato: era l\'unica copia della storia sostituita');
    assert.ok(out.some((m) => m.role === 'user'), 'un turno utente e\' stato toccato');
    assert.match(logDi(sandbox), /SUMMARY GUARD: 1/, 'la guardia non ha contato il riepilogo che ha salvato');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la direzione opposta NON viene dedotta: nessun range inventato per un episodio senza chiusura', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    await apriEChiudi(tools, ctx, 'senza-chiusura');
    const messages = [
      { role: 'user', content: 'apertura' },
      // L'apertura c'e', la chiusura no: un taglio di prefisso non puo' produrre
      // questa disposizione, quindi non ha spiegazione e non si deduce.
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      assistant('INTATTO-1'),
      assistant('INTATTO-2'),
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(contiene(out, 'INTATTO-1') && contiene(out, 'INTATTO-2'),
      'e\' stato inventato un range [start, len-1] per un episodio la cui chiusura non si spiega');
    assert.ok(!/EPISODES deduced: [1-9]/.test(logDi(sandbox)), 'il log dichiara una deduzione che non deve esistere');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('un episodio localizzato vince sulla deduzione dentro il proprio range', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    // 'attraversa' nasce per prima, prima del taglio: dedotta, [0, 5].
    await apriEChiudi(tools, ctx, 'attraversa');
    // 'dentro' nasce dopo il taglio: entrambe le sue ancore sono vive.
    await tools.get('delimiter').execute('call-s2', { action: 'start', name: 'dentro', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e2', { action: 'end', name: 'dentro', description: 'dentro' }, undefined, undefined, ctx);

    const messages = [
      riepilogo(),
      assistant('ATTRAVERSA-1'),
      { role: 'toolResult', toolCallId: 'call-s2', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      assistant('DENTRO-1'),
      { role: 'toolResult', toolCallId: 'call-e2', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(!contiene(out, 'ATTRAVERSA-1'), 'l\'episodio dedotto non e\' stato evacuato');
    // Questo e' il contratto: il range dedotto arriva fino all'indice 5 e copre
    // gli indici 2..4 di 'dentro', ma 'dentro' e' localizzato e aperto DOPO il
    // taglio, quindi scrive per ultimo e rivendica i suoi. Se un giorno il ciclo
    // venisse riordinato in modo ingenuo, questa riga muore.
    assert.ok(contiene(out, 'DENTRO-1'),
      'il range dedotto ha mangiato il contenuto di un episodio localizzato');
  } finally { home.restore(); sandbox.cleanup(); }
});
