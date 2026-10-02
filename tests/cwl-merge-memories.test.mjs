/**
 * THE MERGE OF MEMORIES — several pasts, ONE archive.
 *
 * Why it exists. A memory is forked from another (`cwl_adopt`) and the two branches then live
 * their own lives: one session keeps compressing its own work, the fork keeps compressing its
 * own. Months later the operator has two archives that came from the same conversation, and
 * the only way to use both in one session is to ADOPT them one after the other, which is
 * possible once and impossible twice: the second adoption would overwrite the first. The merge
 * exists to collapse them into a single "super memory" — and it is the only operation in CWL
 * that DELETES what the operator selected, so it is the one with the most to prove.
 *
 * What it must prove, one test per rule:
 *
 *  1. NOTHING IS LOST, NOTHING IS DOUBLED. The same leaf, with the same body, is kept ONCE;
 *     no leaf of any source disappears; and no leaf ends up in two places. This is the rule the
 *     operator stated first, and `placementOf` is the assertion that checks it globally.
 *  2. THE SAME ID WITH DIFFERENT CONTENT KEEPS BOTH. Two branches that edited the same leaf
 *     disagree; the id is not shared, so the extra copy gets a DERIVED id and the conflict is
 *     COUNTED instead of silently overwritten.
 *  3. THE PLACEMENT IS DECIDED BY THE PLACE, NOT BY THE CLOCK. A leaf in a topic in one memory
 *     and loose in the other goes into the TOPIC even when the memory holding the topic is the
 *     older one: a leaf that keeps its place keeps its story. Only when both sides have a REAL
 *     place does the more recent memory win.
 *  4. THE SOURCES ARE DELETED ONLY AFTER THE COPY IS VERIFIED. The new file is written, read
 *     back and checked; only then are the sources removed. And every refusal — the session's own
 *     memory, a live memory, an empty one, a name already taken — must leave the disk untouched.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
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

/** The state file of a session: the key is the session file path, hashed the same way. */
const stateFileOf = (sandbox, sessionFile) =>
  path.join(stateDirOf(sandbox), `${createHash('sha256').update(sessionFile).digest('hex').slice(0, 32)}.json`);

/** Where a merged memory lands: its key is `merge::<name>`, hashed by the same rule. */
const mergedPathOf = (sandbox, name) =>
  path.join(stateDirOf(sandbox), `${createHash('sha256').update(`merge::${name}`).digest('hex').slice(0, 32)}.json`);

const fileOf = (sandbox, file) => path.join(stateDirOf(sandbox), file);

/** A memory as it lies on disk: the fields `listMemories` reads and `planMerge` folds. */
function seedMemory(sandbox, file, name, leaves, extra = {}) {
  const dir = stateDirOf(sandbox);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, file), JSON.stringify({
    version: 4, name, spans: leaves, nodes: [], oldNode: null, ownerPid: 0, savedAt: Date.now(), ...extra,
  }));
}

const leaf = (id, at = 1000, text = `body of ${id}`, micro = `micro of ${id}`) =>
  ({ id, at, summary: text, micro, count: 2 });

const node = (id, leaves, extra = {}) => ({ id, leaves, at: 1000, ...extra });

const readState = (file) => JSON.parse(fs.readFileSync(file, 'utf8'));

/**
 * Where every leaf sits in a merged memory — and a leaf in two places is a DOUBLE.
 * The map is built leaves-first so a duplicate cannot hide: the second time an id shows up
 * the assertion fires with both addresses.
 */
function placementOf(state) {
  const where = new Map();
  for (const nd of state.nodes ?? []) {
    for (const id of nd.leaves ?? []) {
      assert.equal(where.has(id), false, `the leaf ${id} is in ${nd.id} AND in ${where.get(id)}`);
      where.set(id, nd.id);
    }
  }
  for (const sp of state.spans ?? []) {
    if (!where.has(sp.id)) where.set(sp.id, 'loose');
  }
  return where;
}

async function boot(name) {
  const sandbox = makeSandbox({ name: `merge-${name}-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks, commands, notes } = await bootExtension(sandbox, { name: `merge-${name}-${seq}` });
  const sessionFile = path.join(sandbox.dir, 'sessione.jsonl');
  const ctx = sessionCtx(sessionFile);
  return { sandbox, home, tools, hooks, commands, notes, sessionFile, ctx };
}

const mergeCall = (tools, ctx, from, as) =>
  tools.get('cwl_merge_memories').execute('t', { from, as }, undefined, undefined, ctx);

// ---------------------------------------------------------------------------

test('the same leaf with the same body is kept ONCE, and the sources are deleted', async () => {
  const { sandbox, home, tools, ctx } = await boot('sameness');
  try {
    const same = leaf('sp-00000001');
    seedMemory(sandbox, 'a.json', 'mem-a', [same], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [same], { savedAt: 2000 });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.leaves, 1, 'the same leaf was counted twice');
    assert.equal(res.details.conflicts, 0, 'an identical leaf was reported as a conflict');
    assert.deepEqual(res.details.sources.sort(), ['mem-a', 'mem-b']);
    assert.deepEqual(res.details.kept, [], 'a source that should have been deleted was kept');

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.equal(merged.spans.length, 1);
    assert.equal(merged.spans[0].id, 'sp-00000001');
    assert.equal(merged.spans[0].summary, 'body of sp-00000001', 'the body did not survive the merge');
    assert.equal(merged.spans[0].archived, true, 'a merged leaf must be archived: its transcript is not this session');
    assert.equal(merged.ownerPid, 0, 'the super-memory must belong to no living session');
    assert.equal(merged.name, 'unita');
    // The fold is CREATE, not move: the sources are gone only because the merge succeeded.
    assert.equal(fs.existsSync(fileOf(sandbox, 'a.json')), false, 'the source a.json survived');
    assert.equal(fs.existsSync(fileOf(sandbox, 'b.json')), false, 'the source b.json survived');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the same id with a different body keeps BOTH leaves, and the second gets a DERIVED id', async () => {
  const { sandbox, home, tools, ctx } = await boot('conflict');
  try {
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000002', 1000, 'the story as A tells it', 'micro A')], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000002', 1000, 'the story as B tells it', 'micro B')], { savedAt: 2000 });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.leaves, 2, 'a divergent copy was dropped');
    assert.equal(res.details.conflicts, 1, 'the divergence was not counted');
    assert.equal(res.details.renamed, 1);

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    const ids = merged.spans.map((s) => s.id).sort();
    assert.equal(ids.length, 2);
    assert.ok(ids.includes('sp-00000002'), 'the original id was given up');
    const derived = ids.find((id) => id !== 'sp-00000002');
    assert.match(derived, /^sp-[0-9a-f]{8}$/, `the derived id does not look derived: ${derived}`);
    // Both bodies are readable: a renamed leaf is a COPY, never a hole.
    const bodies = new Set(merged.spans.map((s) => s.summary));
    assert.ok(bodies.has('the story as A tells it'));
    assert.ok(bodies.has('the story as B tells it'));
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a different MICRO with the same summary is still a conflict: the leaf is kept twice', async () => {
  const { sandbox, home, tools, ctx } = await boot('micro');
  try {
    const body = 'the same summary in both branches';
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000003', 1000, body, 'micro A')], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000003', 1000, body, 'micro B')], { savedAt: 2000 });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.conflicts, 1, 'the micro is part of the content: a differing micro is a conflict');
    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.equal(new Set(merged.spans.map((s) => s.micro)).size, 2, 'one of the two micros was lost');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('a leaf that is LOOSE in one memory and in a TOPIC in the other goes into the topic', async () => {
  const { sandbox, home, tools, ctx } = await boot('topicwins');
  try {
    const l = leaf('sp-00000004');
    // The memory with the TOPIC is the OLDER one on purpose: the place beats the clock.
    seedMemory(sandbox, 'loose.json', 'mem-loose', [l], { savedAt: 9000 });
    seedMemory(sandbox, 'topic.json', 'mem-topic', [l], {
      savedAt: 1000,
      nodes: [node('nd-topic', ['sp-00000004'], { description: 'a real topic' })],
    });

    const res = await mergeCall(tools, ctx, ['mem-loose', 'mem-topic'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.equal(merged.nodes.length, 1, 'the topic did not survive');
    assert.equal(merged.nodes[0].id, 'nd-topic');
    assert.deepEqual(merged.nodes[0].leaves, ['sp-00000004']);
    assert.equal(merged.nodes[0].description, 'a real topic');
    assert.deepEqual([...placementOf(merged)], [['sp-00000004', 'nd-topic']]);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the SAME topic id in both memories becomes ONE node holding every leaf of both', async () => {
  const { sandbox, home, tools, ctx } = await boot('sametopic');
  try {
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000005', 1000), leaf('sp-00000006', 2000)], {
      savedAt: 2000,
      nodes: [node('nd-shared', ['sp-00000005'], { description: 'descr from A' })],
    });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000005', 1000), leaf('sp-00000006', 2000)], {
      savedAt: 1000,
      nodes: [node('nd-shared', ['sp-00000006'], { description: 'descr from B' })],
    });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.leaves, 2, 'the shared leaves were doubled');
    assert.equal(res.details.conflicts, 0, 'identical leaves of the same topic were called a conflict');
    assert.equal(res.details.nodes, 1, 'one topic id must stay one node');
    assert.equal(res.details.droppedNodes, 0);

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.deepEqual(merged.nodes[0].leaves, ['sp-00000005', 'sp-00000006'], 'the union of the two topics is not there');
    assert.equal(merged.nodes[0].description, 'descr from A', 'the description must come from the most recent memory');
    assert.deepEqual([...placementOf(merged)].sort(), [['sp-00000005', 'nd-shared'], ['sp-00000006', 'nd-shared']]);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('two DIFFERENT topics claiming the same leaf: the most recent memory wins, and the leaf is in ONE node', async () => {
  const { sandbox, home, tools, ctx } = await boot('twotopics');
  try {
    const l = leaf('sp-00000007');
    seedMemory(sandbox, 'a.json', 'mem-a', [l], {
      savedAt: 2000,
      nodes: [node('nd-new', ['sp-00000007'], { description: 'the new home' })],
    });
    seedMemory(sandbox, 'b.json', 'mem-b', [l], {
      savedAt: 1000,
      nodes: [node('nd-old', ['sp-00000007'], { description: 'the old home' })],
    });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.nodes, 1, 'a node that lost every leaf must not survive');
    assert.equal(res.details.droppedNodes, 1, 'the dropped node was not accounted for');

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.equal(merged.nodes[0].id, 'nd-new');
    assert.deepEqual([...placementOf(merged)], [['sp-00000007', 'nd-new']]);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('NOTHING is lost and NOTHING is doubled across four leaves and two topics', async () => {
  const { sandbox, home, tools, ctx } = await boot('union');
  try {
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000011'), leaf('sp-00000012'), leaf('sp-00000013')], {
      savedAt: 2000,
      nodes: [node('nd-a', ['sp-00000012'], { description: 'topic A' })],
    });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000012'), leaf('sp-00000013'), leaf('sp-00000014')], {
      savedAt: 1000,
      nodes: [node('nd-b', ['sp-00000013'], { description: 'topic B' })],
    });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.leaves, 4, 'the union of the sources is not four leaves');

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    const where = placementOf(merged);
    assert.deepEqual([...where.keys()].sort(),
      ['sp-00000011', 'sp-00000012', 'sp-00000013', 'sp-00000014'],
      'the merged memory does not hold exactly the union of the original ids');
    // The PLACE beats the clock, one leaf at a time: 12 is inside a topic in mem-a and loose in
    // mem-b, so it keeps nd-a; 13 is the mirror image and keeps nd-b. Neither is 'the most recent
    // memory's leaf' — recency only decides when BOTH sides offer a real place.
    assert.equal(where.get('sp-00000012'), 'nd-a', 'a leaf that has a topic lost it');
    assert.equal(where.get('sp-00000013'), 'nd-b', 'a leaf that is loose in one memory and in a topic in the other lost the topic');
    assert.equal(where.get('sp-00000011'), 'loose');
    assert.equal(where.get('sp-00000014'), 'loose');
    assert.equal(merged.nodes.length, 2, 'both topics still hold a leaf and must both survive');
    assert.equal(res.details.droppedNodes, 0, 'no node lost every leaf, so none may be reported as dropped');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the PIT comes from the most recent memory and the older synthesis is superseded, not lost', async () => {
  const { sandbox, home, tools, ctx } = await boot('pit');
  try {
    const l = leaf('sp-00000008');
    seedMemory(sandbox, 'a.json', 'mem-a', [l], {
      savedAt: 1000,
      oldNode: { id: 'pit-a', nodes: ['nd-pit'], summary: 'synthesis A', at: 10 },
      nodes: [node('nd-pit', ['sp-00000008'], { description: 'in the pit' })],
    });
    seedMemory(sandbox, 'b.json', 'mem-b', [l], {
      savedAt: 2000,
      oldNode: { id: 'pit-b', nodes: [], summary: 'synthesis B', at: 20 },
    });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));
    assert.equal(res.details.pit, 'pit-b', 'the pit must come from the most recent memory');

    const merged = readState(mergedPathOf(sandbox, 'unita'));
    assert.equal(merged.oldNode.summary, 'synthesis B');
    assert.deepEqual(merged.oldNode.superseded, ['synthesis A'], 'the replaced synthesis was dropped');
    // A pit node of the OTHER memory that survived must still be reachable from the pit.
    assert.deepEqual(merged.oldNode.nodes, ['nd-pit']);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('every REFUSAL leaves the disk exactly as it was', async () => {
  const { sandbox, home, tools, sessionFile, ctx } = await boot('guards');
  const child = spawn(process.execPath, ['-e', 'setTimeout(() => {}, 20000)'], { stdio: 'ignore' });
  try {
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000021')], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000022')], { savedAt: 2000 });
    seedMemory(sandbox, 'live.json', 'mem-live', [leaf('sp-00000023')], { savedAt: 3000, ownerPid: child.pid });
    seedMemory(sandbox, 'empty.json', 'mem-empty', [], { savedAt: 4000 });
    // The session's own file, under the name that `statePath` hashes: merging it would leave the
    // running session writing to a file the merge had just deleted.
    seedMemory(sandbox, path.basename(stateFileOf(sandbox, sessionFile)), 'sessione', [leaf('sp-00000024')], { savedAt: 5000 });

    const before = fs.readdirSync(stateDirOf(sandbox)).sort();
    const refusals = [
      [{ from: [stateFileOf(sandbox, sessionFile), fileOf(sandbox, 'a.json')], as: 'unita' }, 'merge-self'],
      [{ from: ['mem-a', 'mem-b'], as: 'mem-a' }, 'merge-exists'],
      [{ from: ['mem-a', 'mem-a'], as: 'unita' }, 'merge-need-two'],
      [{ from: ['mem-a', 'mem-live'], as: 'unita' }, 'memory-in-use'],
      [{ from: ['mem-a', 'mem-empty'], as: 'unita' }, 'memory-empty'],
      [{ from: ['mem-a', 'non-esiste'], as: 'unita' }, 'memory-not-found'],
      [{ from: ['mem-a', 'mem-b'], as: '   ' }, 'merge-name-required'],
      // A RELATIVE path is NOT a name of a memory: it would have to be anchored to a directory
      // the resolver does not know, so it is read as a name that exists nowhere. Refused, and —
      // like every refusal — it is checked BEFORE any merge is allowed to write.
      [{ from: ['a.json', 'b.json'], as: 'unita-rel' }, 'memory-not-found'],
      // An ABSOLUTE state-file path IS a legitimate way to name a memory, and the only
      // unambiguous one when two memories share a name. This is the LAST entry on purpose: it is
      // the only one that writes, so every refusal above is measured against the pristine disk.
      [{ from: [fileOf(sandbox, 'a.json'), fileOf(sandbox, 'b.json')], as: 'unita' }, null],
    ];
    for (const [args, error] of refusals) {
      const res = await mergeCall(tools, ctx, args.from, args.as);
      if (error === null) {
        assert.equal(res.details.ok, true, `merging by PATH was refused: ${JSON.stringify(res.details)}`);
        continue;
      }
      assert.equal(res.details.ok, false, `${error} was not refused: ${JSON.stringify(args)}`);
      assert.equal(res.details.error, error, `unexpected refusal for ${JSON.stringify(args)}`);
      assert.deepEqual(fs.readdirSync(stateDirOf(sandbox)).sort(), before,
        `${error} touched the disk: nothing may be written when a merge is refused`);
    }

    // The one merge that went through is the last one: the sources it used are gone, and the
    // three memories that no refusal had the right to delete are all still there.
    for (const f of ['live.json', 'empty.json', 'a.json', 'b.json']) {
      const shouldBeThere = f === 'live.json' || f === 'empty.json';
      assert.equal(fs.existsSync(fileOf(sandbox, f)), shouldBeThere, `${f} is in the wrong state`);
    }
  } finally { child.kill(); home.restore(); sandbox.cleanup(); }
});

test('after the merge the super-memory is a memory like any other: it can be ADOPTED', async () => {
  const { sandbox, home, tools, hooks, sessionFile, ctx } = await boot('adoptable');
  try {
    const l = leaf('sp-00000031');
    seedMemory(sandbox, 'a.json', 'mem-a', [l], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000032')], { savedAt: 2000 });

    const res = await mergeCall(tools, ctx, ['mem-a', 'mem-b'], 'unita');
    assert.equal(res.details.ok, true, JSON.stringify(res.details));

    // A second session adopts what the merge produced — the whole point of the operation.
    const other = sessionCtx(sessionFile);
    await hooks.get('session_start')({}, other);
    const adopted = await tools.get('cwl_adopt').execute('t', { from: 'unita', as: 'ramo' }, undefined, undefined, other);
    assert.equal(adopted.details.ok, true, `the super-memory was not adoptable: ${JSON.stringify(adopted.details)}`);

    const ramo = readState(stateFileOf(sandbox, sessionFile));
    assert.equal(ramo.spans.length, 2, 'the adopted super-memory did not bring its leaves');
    assert.ok(ramo.spans.every((s) => s.archived === true), 'an adopted leaf must arrive archived');
    assert.deepEqual(ramo.spans.map((s) => s.id).sort(), ['sp-00000031', 'sp-00000032']);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('/cwl_merge is registered, and the picker is what decides', async () => {
  const { sandbox, home, commands, sessionFile } = await boot('command');
  try {
    seedMemory(sandbox, 'a.json', 'mem-a', [leaf('sp-00000041')], { savedAt: 1000 });
    seedMemory(sandbox, 'b.json', 'mem-b', [leaf('sp-00000042')], { savedAt: 2000 });

    assert.ok(commands.has('cwl_merge'), '/cwl_merge is NOT registered');
    const def = commands.get('cwl_merge');
    assert.equal(typeof def.handler, 'function');
    assert.ok(typeof def.description === 'string' && def.description.length > 0, 'the command has no description');

    const notes = [];
    const offered = [];
    const ui = {
      // Always the first of what is LEFT: the newest memory first, then whatever remains. A
      // cursor into the ORIGINAL list would run off the end of the shrinking pool.
      select: async (_title, options) => { offered.push(options); return options[0]; },
      confirm: async () => true,
      input: async () => 'unita',
      notify: (text, kind) => notes.push({ text, kind }),
    };
    const ctx = { cwd: '/tmp/progetto', hasUI: true, sessionManager: { getSessionFile: () => sessionFile }, ui };
    await def.handler('', ctx);

    assert.equal(offered.length, 2, 'the command did not ask for a second memory');
    assert.match(offered[0].join(' '), /mem-b/, 'the newest memory is not offered first');
    assert.ok(!offered[1].some((o) => o.includes('mem-b')), 'the memory already picked was offered again');
    assert.ok(!offered[0].some((o) => o.includes('LIVE')), 'a live memory must not be offered');

    assert.equal(fs.existsSync(mergedPathOf(sandbox, 'unita')), true, 'the command wrote no memory');
    assert.equal(fs.existsSync(fileOf(sandbox, 'a.json')), false);
    assert.equal(fs.existsSync(fileOf(sandbox, 'b.json')), false);
    assert.ok(notes.some((n) => n.kind === 'info' && n.text.includes('unita')), `no confirmation was shown: ${JSON.stringify(notes)}`);

    // A cancelled picker writes NOTHING: the decision belongs to the operator.
    seedMemory(sandbox, 'c.json', 'mem-c', [leaf('sp-00000043')], { savedAt: 3000 });
    seedMemory(sandbox, 'd.json', 'mem-d', [leaf('sp-00000044')], { savedAt: 4000 });
    const cancelNotes = [];
    const cancelCtx = {
      cwd: '/tmp/progetto', hasUI: true, sessionManager: { getSessionFile: () => sessionFile },
      ui: { select: async () => undefined, confirm: async () => true, input: async () => 'mai', notify: (t, k) => cancelNotes.push({ t, k }) },
    };
    await def.handler('', cancelCtx);
    assert.equal(fs.existsSync(mergedPathOf(sandbox, 'mai')), false, 'an abandoned picker wrote a memory');
    assert.equal(fs.existsSync(fileOf(sandbox, 'c.json')), true, 'an abandoned picker deleted a source');
    assert.equal(fs.existsSync(fileOf(sandbox, 'd.json')), true);
    assert.ok(cancelNotes.length > 0, 'the cancellation was silent');
  } finally { home.restore(); sandbox.cleanup(); }
});
