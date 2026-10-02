import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

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
const stateFileOf = (sandbox, sessionFile) =>
  path.join(stateDirOf(sandbox), `${createHash('sha256').update(sessionFile).digest('hex').slice(0, 32)}.json`);

function seedMemory(sandbox, file, name, leaves, extra = {}) {
  const dir = stateDirOf(sandbox);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, file), JSON.stringify({
    version: 4, name, spans: leaves, nodes: [], oldNode: null, ownerPid: 0, savedAt: Date.now(), ...extra,
  }));
}

const leaf = (id, at = 1000, text = `body of ${id}`, micro = `micro of ${id}`) =>
  ({ id, at, summary: text, micro, count: 2 });

const readState = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));

async function boot(name) {
  const sandbox = makeSandbox({ name: `absorb-${name}-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks, commands, notes } = await bootExtension(sandbox, { name: `absorb-${name}-${seq}` });
  const sessionFile = path.join(sandbox.dir, 'sessione.jsonl');
  const ctx = sessionCtx(sessionFile);
  return { sandbox, home, tools, hooks, commands, notes, sessionFile, ctx };
}

const absorbCall = (tools, ctx, from) =>
  tools.get('cwl_absorb_memories').execute('t', { from }, undefined, undefined, ctx);

test('cwl_absorb_memories — brings new leaves, upgrades micro, deduplicates by body, and leaves sources untouched', async () => {
  const { sandbox, home, tools, ctx, sessionFile } = await boot('basic');
  try {
    seedMemory(sandbox, 'alpha.json', 'alpha', [
      leaf('sp-old', 100, 'shared body text', 'old micro wording'),
      leaf('sp-new', 200, 'brand new body from alpha', 'micro alpha'),
    ], {
      nodes: [{ id: 'nd-alpha', leaves: ['sp-alpha-topic'] }],
      oldNode: { id: 'pit-alpha', nodes: ['nd-alpha'], summary: 'alpha synthesis', at: 50 },
    });

    // Seed session's own state file with one local leaf having the same body but a newer micro
    const stateFile = stateFileOf(sandbox, sessionFile);
    fs.mkdirSync(path.dirname(stateFile), { recursive: true });
    fs.writeFileSync(stateFile, JSON.stringify({
      version: 4,
      name: 'current',
      spans: [leaf('sp-local', 150, 'shared body text', 'NEWER micro wording')],
      nodes: [],
      oldNode: null,
      ownerPid: process.pid,
      savedAt: Date.now(),
    }));

    const res = await absorbCall(tools, ctx, ['alpha']);
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.added, 1, 'sp-new was added');

    const saved = readState(stateFile);
    assert.equal(saved.spans.length, 2);
    const shared = saved.spans.find((s) => s.summary === 'shared body text');
    assert.ok(shared);
    assert.equal(shared.micro, 'NEWER micro wording', 'newer micro won');

    // Sources must be UNTOUCHED on disk
    const alphaPath = path.join(stateDirOf(sandbox), 'alpha.json');
    assert.equal(fs.existsSync(alphaPath), true, 'source memory file must remain intact');
    const sourceData = readState(alphaPath);
    assert.equal(sourceData.spans.length, 2, 'source spans were not deleted');

    // Repeated absorb should be a no-op (added = 0)
    const res2 = await absorbCall(tools, ctx, ['alpha']);
    assert.equal(res2.details.ok, true);
    assert.equal(res2.details.added, 0);
  } finally { home.restore(); sandbox.cleanup(); }
});
