/**
 * Regressione: lo stato di sessione sopravvive al riavvio, e due sessioni
 * diverse non si sovrascrivono a vicenda.
 *
 * Perche' esiste. Tre difetti dell'area "session state":
 *
 * 1. `sessionKey` era `${cwd}::${sid}`. Senza id di sessione diventava `::` o
 *    la costante `default`, quindi DUE sessioni anonime nella stessa cartella
 *    condividevano un unico stato.
 * 2. `session_start` faceva `states.set(key, newState())` senza guardare se lo
 *    stato esisteva: il grafo degli episodi veniva azzerato a ogni avvio.
 * 3. Lo stato viveva solo in RAM. Il grafo degli episodi e le compressioni
 *    tracciate con `cwl_compress` sono DECISIONI dell'agente, non dati
 *    ricalcolabili dal transcript: persi al riavvio, il lavoro era perso.
 *
 * Il test guida i veri hook dell'estensione (session_start / delimiter /
 * cwl_compress / context / turn_end) e riavvia il modulo da capo, come farebbe
 * un riavvio di Pi.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { makeSandbox, bootExtension, withHome, sessionCtx, anonymousCtx, status } from './_helpers.mjs';

/** Avvia una nuova istanza indipendente dell'estensione sullo stesso sandbox. */
const reboot = (sandbox, tag) => bootExtension(sandbox, { name: tag });

/** Hash del testo di un messaggio, come lo calcola hashText() nell'estensione. */
const hashOf = (text) => createHash('sha256').update(text).digest('hex').slice(0, 12);

const msg = (text, role = 'assistant') => ({ role, content: text, timestamp: 1 });

test('il grafo degli episodi sopravvive a un riavvio', async () => {
  const sandbox = makeSandbox({ name: 'state-graph' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));

    // --- Prima esecuzione: apri e chiudi un episodio.
    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.tools.get('delimiter').execute('tc1',
      { action: 'start', name: 'esplorazione-1', type: 'expl' }, undefined, undefined, session);
    await run1.tools.get('delimiter').execute('tc2',
      { action: 'end', name: 'esplorazione-1', description: 'imparato X' }, undefined, undefined, session);
    await run1.hooks.get('turn_end')({}, session);
    assert.equal((await status(run1.tools, session)).details.total, 1, 'prima del riavvio: 1 episodio');

    // --- Seconda esecuzione: modulo nuovo, stessa HOME, stessa sessione.
    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const after = await status(run2.tools, session);
    assert.equal(after.details.total, 1, "dopo il riavvio l'episodio deve essere ritrovato, non perso");
    assert.equal(after.details.closed, 1, "l'episodio deve restare chiuso, com'era prima del riavvio");
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('le compressioni tracciate sopravvivono a un riavvio', async () => {
  const sandbox = makeSandbox({ name: 'state-span' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
    // Entrambi gli estremi sono messaggi ASSISTANT: i messaggi `user` sono
    // inviolabili per design (Principio 3), quindi un range user->user
    // conserverebbe gli originali e si limiterebbe ad AGGIUNGERE il riepilogo.
    // Con due assistant lo span viene davvero sostituito e la prova e' netta.
    const all = [msg('primo blocco di lavoro'), msg('secondo blocco di lavoro')];

    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    // Il context hook registra gli hash dei messaggi presenti: senza quelli
    // cwl_compress rifiuterebbe l'hash come sconosciuto.
    await run1.hooks.get('context')({ messages: all }, session);

    const res = await run1.tools.get('cwl_compress').execute('tc3', {
      startHash: hashOf(all[0].content),
      endHash: hashOf(all[1].content),
      summary: 'riepilogo della coppia',
    }, undefined, undefined, session);
    assert.equal(res.details?.ok, true, `la compressione deve essere accettata (${res.content[0].text})`);
    assert.equal(res.details?.spans, 1);

    // --- Riavvio.
    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const messages = [msg('primo blocco di lavoro'), msg('secondo blocco di lavoro')];
    const out = await run2.hooks.get('context')({ messages }, session);
    assert.ok(out?.messages, 'la compressione ripresa deve essere applicata al contesto');
    assert.ok(out.messages.length < messages.length,
      `i due messaggi devono essere sostituiti dal riepilogo: ${out.messages.length} vs ${messages.length}`);
    const joined = JSON.stringify(out.messages);
    assert.match(joined, /riepilogo della coppia/, 'il testo del riepilogo deve essere nel contesto');
    assert.doesNotMatch(joined, /primo blocco di lavoro/, 'i testi originali devono essere stati sostituiti');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('una compressione revocata non risorge dal disco', async () => {
  const sandbox = makeSandbox({ name: 'state-revoke' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
    const pair = [msg('alfa'), msg('beta')];

    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.hooks.get('context')({ messages: pair }, session);

    const tool = run1.tools.get('cwl_compress');
    const args = { startHash: hashOf('alfa'), endHash: hashOf('beta'), summary: 'S' };
    await tool.execute('t1', args, undefined, undefined, session);
    const revoke = await tool.execute('t2', args, undefined, undefined, session);
    assert.equal(revoke.details?.revoked, true, 'la seconda chiamata deve revocare');

    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const out = await run2.hooks.get('context')({ messages: pair }, session);
    // Nessuno span: il contesto non deve essere toccato (siamo sotto soglia).
    assert.ok(!out?.messages || out.messages.length === pair.length,
      'una compressione revocata non deve essere riapplicata dopo il riavvio');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('due sessioni senza id non condividono lo stato', async () => {
  const sandbox = makeSandbox({ name: 'state-anon' });
  const home = withHome(sandbox.dir);
  try {
    const { hooks, tools } = await bootExtension(sandbox);
    const a = anonymousCtx();
    const b = anonymousCtx();

    await hooks.get('session_start')({}, a);
    await hooks.get('session_start')({}, b);

    // Un episodio aperto nella sessione A non deve comparire nella B.
    await tools.get('delimiter').execute('tc1',
      { action: 'start', name: 'solo-di-a', type: 'expl' }, undefined, undefined, a);
    assert.equal((await status(tools, a)).details.total, 1, 'A vede il proprio episodio');
    assert.equal((await status(tools, b)).details.total, 0, 'B NON deve vedere lo stato di A');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('i file di stato vengono potati quando sono vecchi', async () => {
  const sandbox = makeSandbox({ name: 'state-prune' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.tools.get('delimiter').execute('tc1',
      { action: 'start', name: 'e', type: 'expl' }, undefined, undefined, session);
    await run1.hooks.get('turn_end')({}, session);

    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    const files = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(files.length, 1, 'un file di stato per la sessione');

    // Invecchia il file oltre la soglia di ritenzione.
    const old = new Date(Date.now() - 20 * 24 * 60 * 60 * 1000);
    for (const f of files) fs.utimesSync(path.join(stateDir, f), old, old);

    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const left = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(left.length, 0, 'un file di stato vecchio deve essere rimosso');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('una sessione che non usa CWL non scrive nulla su disco', async () => {
  const sandbox = makeSandbox({ name: 'state-idle' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
    const { hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, session);
    await hooks.get('context')({ messages: [msg('ciao')] }, session);
    await hooks.get('turn_end')({}, session);

    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    const files = fs.existsSync(stateDir)
      ? fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'))
      : [];
    assert.equal(files.length, 0, 'senza episodi ne\' compressioni non deve esserci stato da salvare');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('uno stato corrotto non impedisce l\'avvio', async () => {
  const sandbox = makeSandbox({ name: 'state-corrupt' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    fs.mkdirSync(stateDir, { recursive: true });

    // Scrive uno stato illeggibile con lo stesso nome che userebbe l'estensione.
    const key = session.sessionManager.getSessionFile();
    const name = `${createHash('sha256').update(key).digest('hex').slice(0, 32)}.json`;
    fs.writeFileSync(path.join(stateDir, name), '{ troncato a meta');

    const { hooks, tools } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, session);
    const st = await status(tools, session);
    assert.equal(st.details.total, 0, 'un file corrotto deve degradare a stato vuoto, non far fallire l\'avvio');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
