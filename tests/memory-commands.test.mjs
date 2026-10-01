/**
 * THE SLASH COMMANDS — the fork in the OPERATOR's hands.
 *
 * Why it exists. The tools `cwl_memories` and `cwl_adopt` are callable by the MODEL, and
 * forking a memory is a decision about which past to continue: it belongs to whoever sits
 * at the keyboard. The two slash commands give that decision back to the operator
 * (`/cwl_memories` to look, `/cwl_adopt` to pick and name), and these tests pin the three
 * properties that make them trustworthy:
 *
 *  1. THEY EXIST. A command that is not registered is a command the operator will look for
 *     in the slash menu and not find.
 *  2. THE PICK IS WHAT IS FORKED, AND THE SOURCE IS NEVER TOUCHED. The command must resolve
 *     the same way the tool does — and a cancelled dialog must write NOTHING, because a
 *     half-adoption would silently replace the session's memory.
 *  3. A LIVE MEMORY IS SHOWN BUT NEVER OFFERED. A memory whose owning session is still
 *     running keeps writing its own copy; forking it would give two writers to one past.
 *     The listing marks it `(LIVE)`, and the picker leaves it out.
 *
 * The test drives the handler with a scripted `ui`, which is exactly the contract Pi gives
 * a command: `select` returns the chosen string or `undefined` on Esc, `input` the typed
 * text or `undefined`, `notify` is fire-and-forget.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { makeSandbox, bootExtension, withHome } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

const stateDirOf = (sandbox) => path.join(sandbox.dir, '.pi', 'cwl', 'state');

/** The state file of a session: the key is the session file path, hashed the same way. */
const stateFileOf = (sandbox, sessionFile) =>
  path.join(stateDirOf(sandbox), `${createHash('sha256').update(sessionFile).digest('hex').slice(0, 32)}.json`);

/** A memory as it lies on disk: the fields `listMemories` reads and `performAdopt` copies. */
function seedMemory(sandbox, file, name, leaves, extra = {}) {
  const dir = stateDirOf(sandbox);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, file), JSON.stringify({
    version: 4, name, spans: leaves, nodes: [], oldNode: null, savedAt: Date.now(), ...extra,
  }));
}

const leaf = (id) => ({ id, at: Date.now(), summary: `summary of ${id}`, micro: `micro of ${id}`, count: 2 });

/** A command context: `hasUI` true and a scripted `ui`, as a real command receives. */
function commandCtx({ sessionFile, notes, select = () => undefined, input = () => undefined }) {
  return {
    cwd: '/tmp/progetto',
    hasUI: true,
    sessionManager: { getSessionFile: () => sessionFile },
    ui: {
      select: async (title, options) => select(title, options),
      input: async (title, placeholder) => input(title, placeholder),
      notify: (text, kind) => notes.push({ text, kind }),
    },
  };
}

async function boot(name) {
  const sandbox = makeSandbox({ name: `cmd-${name}-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks, commands } = await bootExtension(sandbox);
  const sessionFile = path.join(sandbox.dir, 'sessione.jsonl');
  return { sandbox, home, tools, hooks, commands, sessionFile };
}

test('the two commands are registered, each with a description', async () => {
  const { sandbox, home, commands } = await boot('reg');
  try {
    for (const name of ['cwl_memories', 'cwl_adopt']) {
      assert.ok(commands.has(name), `${name} is registered`);
      assert.equal(typeof commands.get(name).handler, 'function', `${name} has a handler`);
      assert.equal(typeof commands.get(name).description, 'string', `${name} has a description`);
      assert.ok(commands.get(name).description.length > 0, `${name}'s description is not empty`);
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_adopt forks the picked memory under the name the operator typed', async () => {
  const { sandbox, home, hooks, commands, sessionFile } = await boot('adopt');
  try {
    seedMemory(sandbox, 'aaaa.json', 'mem-a', [leaf('L1'), leaf('L2')]);
    const sourcePath = path.join(stateDirOf(sandbox), 'aaaa.json');
    const sourceBefore = fs.readFileSync(sourcePath, 'utf8');
    const notes = [];
    const ctx = commandCtx({ sessionFile, notes, select: () => 'mem-a', input: () => 'my-copy' });
    await hooks.get('session_start')({}, ctx);
    await commands.get('cwl_adopt').handler('', ctx);

    const forked = JSON.parse(fs.readFileSync(stateFileOf(sandbox, sessionFile), 'utf8'));
    assert.equal(forked.name, 'my-copy', 'the fork carries the typed name');
    assert.equal(forked.spans.length, 2, 'both leaves were copied');
    assert.ok(forked.spans.every((s) => s.archived === true), 'every forked leaf is archived');
    assert.equal(fs.readFileSync(sourcePath, 'utf8'), sourceBefore, 'the source memory is never touched');
    assert.match(notes.at(-1).text, /my-copy/, 'the operator is told what was created');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cancelling the pick writes nothing', async () => {
  const { sandbox, home, hooks, commands, sessionFile } = await boot('cancel');
  try {
    seedMemory(sandbox, 'aaaa.json', 'mem-a', [leaf('L1')]);
    const notes = [];
    const ctx = commandCtx({ sessionFile, notes, select: () => undefined });
    await hooks.get('session_start')({}, ctx);
    const before = fs.readdirSync(stateDirOf(sandbox)).length;
    await commands.get('cwl_adopt').handler('', ctx);
    assert.equal(fs.readdirSync(stateDirOf(sandbox)).length, before, 'no state file was written');
    assert.match(notes.at(-1).text, /Nothing was adopted/, 'the operator is told nothing happened');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a LIVE memory is shown but never offered for adoption', async () => {
  const { sandbox, home, hooks, commands, sessionFile } = await boot('live');
  const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 20000)'], { stdio: 'ignore' });
  try {
    seedMemory(sandbox, 'aaaa.json', 'mem-a', [leaf('L1')]);
    seedMemory(sandbox, 'live.json', 'mem-live', [leaf('L1')], { ownerPid: child.pid });
    const notes = [];
    let offered = null;
    const ctx = commandCtx({
      sessionFile, notes,
      select: (_title, options) => { offered = options; return 'mem-a'; },
      input: () => '',
    });
    await hooks.get('session_start')({}, ctx);

    // The listing shows it, marked.
    await commands.get('cwl_memories').handler('', ctx);
    assert.match(notes.at(-1).text, /mem-live \(LIVE\)/, 'the listing marks the live memory');
    assert.match(notes.at(-1).text, /mem-a/, 'the listing shows the free memory');

    // The picker leaves it out, and an empty name keeps the default.
    await commands.get('cwl_adopt').handler('', ctx);
    assert.ok(offered.includes('mem-a'), 'the adoptable memory is offered');
    assert.ok(!offered.some((n) => n.includes('mem-live')), 'the LIVE memory is not offered');
    const forked = JSON.parse(fs.readFileSync(stateFileOf(sandbox, sessionFile), 'utf8'));
    assert.equal(forked.name, 'mem-a--fork', 'an empty name keeps the default');
  } finally {
    child.kill();
    home.restore(); sandbox.cleanup();
  }
});
