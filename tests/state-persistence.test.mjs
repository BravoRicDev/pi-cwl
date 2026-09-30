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
 * turn_end) e riavvia il modulo da capo, come farebbe un riavvio di Pi.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from '/home/riccardo/.hermes/lsp/node_modules/typescript/lib/typescript.js';

const root = path.resolve(import.meta.dirname, '..');
const TYPEBOX = '/home/riccardo/.hermes/node/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/typebox/build/index.mjs';

/**
 * Prepara una HOME temporanea e ci scrive una COPIA del transpilato.
 * Copie diverse (stesso path sorgente, nomi file diversi) danno istanze di
 * modulo indipendenti: e' cosi' che si simula un riavvio del processo.
 */
function setupHome() {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'cwl-state-'));
  const source = fs.readFileSync(path.join(root, 'index.ts'), 'utf8');
  const js = ts
    .transpileModule(source, {
      compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
    })
    .outputText.replace("from 'typebox'", `from '${pathToFileURL(TYPEBOX).href}'`);
  fs.copyFileSync(path.join(root, 'config.json'), path.join(temp, 'config.json'));
  return { temp, js };
}

/** Istanza fresca dell'estensione, con gli hook registrati e catturati. */
async function boot(temp, js, name) {
  const file = path.join(temp, `${name}.mjs`);
  fs.writeFileSync(file, js);
  const { default: extension } = await import(pathToFileURL(file).href);
  const tools = new Map();
  const hooks = new Map();
  const pi = {
    on(event, handler) { hooks.set(event, handler); },
    registerTool(tool) { tools.set(tool.name, tool); },
    registerCommand() {},
    sendMessage() {},
  };
  extension(pi);
  return { tools, hooks };
}

/** Contesto di sessione: identita' stabile fornita dal transcript. */
const ctxWith = (sessionFile, cwd = '/tmp/progetto') => ({
  cwd,
  hasUI: false,
  sessionManager: {
    getSessionFile: () => sessionFile,
    getSessionId: () => 'non-usato-quando-c-e-il-file',
  },
});

/** Contesto senza alcun identificativo: e' il caso che collassava su "default". */
const ctxAnonymous = () => ({ cwd: '/tmp/progetto', hasUI: false });

/**
 * Stato della sessione letto dai `details` del tool: i numeri sono
 * locale-indipendenti, il testo no (con LANG=it_IT.UTF-8 il modulo risponde
 * in italiano, e asserire stringhe inglesi fallirebbe).
 */
async function status(tools, ctx) {
  const res = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
  return { text: res.content.map((c) => c.text).join('\n'), details: res.details };
}

test('il grafo degli episodi sopravvive a un riavvio', async () => {
  const { temp, js } = setupHome();
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const session = ctxWith(path.join(temp, 'sessione.jsonl'));

    // --- Prima esecuzione: apri e chiudi un episodio.
    const boot1 = await boot(temp, js, 'run1');
    await boot1.hooks.get('session_start')({}, session);
    await boot1.tools.get('delimiter').execute('tc1', { action: 'start', name: 'esplorazione-1', type: 'expl' }, undefined, undefined, session);
    await boot1.tools.get('delimiter').execute('tc2', { action: 'end', name: 'esplorazione-1', description: 'imparato X' }, undefined, undefined, session);
    await boot1.hooks.get('turn_end')({}, session);
    assert.equal((await status(boot1.tools, session)).details.total, 1, 'prima del riavvio: 1 episodio');

    // --- Seconda esecuzione: modulo nuovo, stessa HOME, stessa sessione.
    const boot2 = await boot(temp, js, 'run2');
    await boot2.hooks.get('session_start')({}, session);
    const after = await status(boot2.tools, session);
    assert.equal(after.details.total, 1, 'dopo il riavvio l\'episodio deve essere ritrovato, non perso');
    assert.equal(after.details.closed, 1, 'l\'episodio deve restare chiuso, com\'era prima del riavvio');
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('le compressioni tracciate sopravvivono a un riavvio', async () => {
  const { temp, js } = setupHome();
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const session = ctxWith(path.join(temp, 'sessione.jsonl'));
    // Entrambi gli estremi sono messaggi ASSISTANT: i messaggi `user` sono
    // inviolabili per design (Principio 3), quindi un range user->user
    // conserverebbe gli originali e si limiterebbe ad AGGIUNGERE il riepilogo
    // (3 messaggi invece di 2). Con due assistant lo span viene sostituito
    // davvero e la prova e' netta.
    const msg = (t, role = 'assistant') => ({ role, content: t, timestamp: 1 });

    const boot1 = await boot(temp, js, 'run1');
    await boot1.hooks.get('session_start')({}, session);
    // Il context hook registra gli hash dei messaggi presenti: senza quelli
    // cwl_compress rifiuterebbe l'hash come sconosciuto.
    const all = [msg('primo blocco di lavoro'), msg('secondo blocco di lavoro')];
    await boot1.hooks.get('context')({ messages: all }, session);

    // L'hash e' lo sha256 del testo troncato a 12 hex, come in hashText().
    const { createHash } = await import('node:crypto');
    const h = (s) => createHash('sha256').update(s).digest('hex').slice(0, 12);

    const resCompress = await boot1.tools.get('cwl_compress').execute(
      'tc3',
      { startHash: h(all[0].content), endHash: h(all[1].content), summary: 'riepilogo della coppia' },
      undefined, undefined, session,
    );
    assert.equal(resCompress.details?.ok, true, `la compressione deve essere accettata (${resCompress.content[0].text})`);
    assert.equal(resCompress.details?.spans, 1);

    // --- Riavvio.
    const boot2 = await boot(temp, js, 'run2');
    await boot2.hooks.get('session_start')({}, session);
    const messages = [msg('primo blocco di lavoro'), msg('secondo blocco di lavoro')];
    const out = await boot2.hooks.get('context')({ messages }, session);
    assert.ok(out?.messages, 'la compressione ripresa deve essere applicata al contesto');
    assert.ok(
      out.messages.length < messages.length,
      `i due messaggi devono essere sostituiti dal riepilogo: ${out.messages.length} vs ${messages.length}`,
    );
    const joined = JSON.stringify(out.messages);
    assert.match(joined, /riepilogo della coppia/, 'il testo del riepilogo deve essere nel contesto');
    assert.doesNotMatch(joined, /primo blocco di lavoro/, 'i testi originali devono essere stati sostituiti');
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('una compressione revocata non risorge dal disco', async () => {
  const { temp, js } = setupHome();
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const session = ctxWith(path.join(temp, 'sessione.jsonl'));
    const msg = (t) => ({ role: 'user', content: t, timestamp: 1 });
    const boot1 = await boot(temp, js, 'run1');
    await boot1.hooks.get('session_start')({}, session);
    await boot1.hooks.get('context')({ messages: [msg('alfa'), msg('beta')] }, session);

    const { createHash } = await import('node:crypto');
    const h = (s) => createHash('sha256').update(s).digest('hex').slice(0, 12);
    const tool = boot1.tools.get('cwl_compress');
    await tool.execute('t1', { startHash: h('alfa'), endHash: h('beta'), summary: 'S' }, undefined, undefined, session);
    const revoke = await tool.execute('t2', { startHash: h('alfa'), endHash: h('beta'), summary: 'S' }, undefined, undefined, session);
    assert.equal(revoke.details?.revoked, true, 'la seconda chiamata deve revocare');

    const boot2 = await boot(temp, js, 'run2');
    await boot2.hooks.get('session_start')({}, session);
    const messages = [msg('alfa'), msg('beta')];
    const out = await boot2.hooks.get('context')({ messages }, session);
    // Nessuno span: il contesto non deve essere toccato (l'eviction non scatta
    // perche' siamo sotto soglia).
    assert.ok(
      !out?.messages || out.messages.length === messages.length,
      'una compressione revocata non deve essere riapplicata dopo il riavvio',
    );
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('due sessioni senza id non condividono lo stato', async () => {
  const { temp, js } = setupHome();
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const boot1 = await boot(temp, js, 'run1');
    const a = ctxAnonymous();
    const b = ctxAnonymous();

    await boot1.hooks.get('session_start')({}, a);
    await boot1.hooks.get('session_start')({}, b);

    // Un episodio aperto nella sessione A non deve comparire nella B.
    await boot1.tools.get('delimiter').execute('tc1', { action: 'start', name: 'solo-di-a', type: 'expl' }, undefined, undefined, a);
    assert.equal((await status(boot1.tools, a)).details.total, 1, 'A vede il proprio episodio');
    assert.equal((await status(boot1.tools, b)).details.total, 0, 'B NON deve vedere lo stato di A');
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});

test('i file di stato vengono potati quando sono vecchi', async () => {
  const { temp, js } = setupHome();
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const session = ctxWith(path.join(temp, 'sessione.jsonl'));
    const boot1 = await boot(temp, js, 'run1');
    await boot1.hooks.get('session_start')({}, session);
    await boot1.tools.get('delimiter').execute('tc1', { action: 'start', name: 'e', type: 'expl' }, undefined, undefined, session);
    await boot1.hooks.get('turn_end')({}, session);

    const stateDir = path.join(temp, '.pi', 'cwl', 'state');
    const files = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(files.length, 1, 'un file di stato per la sessione');

    // Invecchia il file oltre la soglia di ritenzione.
    const old = new Date(Date.now() - 20 * 24 * 60 * 60 * 1000);
    for (const f of files) fs.utimesSync(path.join(stateDir, f), old, old);

    const boot2 = await boot(temp, js, 'run2');
    await boot2.hooks.get('session_start')({}, session);
    const left = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(left.length, 0, 'un file di stato vecchio deve essere rimosso');
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
