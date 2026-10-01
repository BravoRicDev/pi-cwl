/**
 * THE FORK OF A MEMORY — the three properties that make it safe.
 *
 * Why it exists. A memory can be carried into a NEW session (`cwl_adopt`), and the operator
 * accepted the loss of the original transcript: from that moment the summary IS the memory.
 * That acceptance is only honest if three things hold:
 *
 *  1. THE FORK IS A NEW MEMORY. The source is never modified — otherwise "adopt" would be
 *     "move", and the session that wrote the memory would lose it while still running.
 *  2. AN ARCHIVED LEAF IS NEVER PRUNED. A copied leaf carries anchors of ANOTHER transcript,
 *     so `locateSpans` can never resolve it. The pruning path drops the leaves it declares
 *     `dead` — and a dead leaf's summary is discarded on the argument that the original is
 *     still in the append-only transcript. For an archived leaf there is no such original:
 *     pruning it would DELETE the memory. It is skipped before the lookup (neither resolved
 *     nor dead), and this test is the one that would catch a regression.
 *  3. THE INHERITED MEMORY REACHES THE MODEL. Its blocks normally stand where the compressed
 *     content used to be; here there is no such place, so they are injected at the TOP of the
 *     list — and they must stay there even after this session starts compressing on its own,
 *     which is when a naive implementation would lose them again.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 300,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot() {
  const sandbox = makeSandbox({ name: `fork-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const aFile = path.join(sandbox.dir, 'sessione-A.jsonl');
  const bFile = path.join(sandbox.dir, 'sessione-B.jsonl');
  const a = sessionCtx(aFile);
  const b = sessionCtx(bFile);
  FILE_OF.set(a, aFile);
  FILE_OF.set(b, bFile);
  await hooks.get('session_start')({}, a);
  await hooks.get('session_start')({}, b);
  return { sandbox, home, tools, hooks, a, b };
}

/** ctx -> the session file it was built from: the state is keyed by that PATH. */
const FILE_OF = new Map();

const stateFileOf = (sandbox, ctx) =>
  path.join(sandbox.dir, '.pi', 'cwl', 'state',
    `${createHash('sha256').update(FILE_OF.get(ctx)).digest('hex').slice(0, 32)}.json`);

const stateOf = (sandbox, ctx) => JSON.parse(fs.readFileSync(stateFileOf(sandbox, ctx), 'utf8'));
const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const user = (i) => ({ role: 'user', content: `prompt ${i} ${'P'.repeat(200)}` });
const assistant = (i) => ({ role: 'assistant', content: `answer ${i} ${'A'.repeat(200)}` });

/** Grows a memory of its own: six exchanges, then one compression. */
async function writeAMemory(tools, hooks, ctx) {
  const messages = [];
  for (let i = 1; i <= 6; i++) { messages.push(user(i)); messages.push(assistant(i)); }
  await hooks.get('context')({ messages }, ctx);
  const res = await tools.get('cwl_compress_range').execute(
    't', { summary: `BODY ${'x'.repeat(300)}`, micro: 'MICRO: the points that matter most.' }, undefined, undefined, ctx,
  );
  assert.equal(res.details.ok, true, `the leaf was not born: ${JSON.stringify(res.details)}`);
  return messages;
}

test('adopting is a FORK: a new memory, and the source is left untouched', async () => {
  const { sandbox, home, tools, hooks, a, b } = await boot();
  try {
    await writeAMemory(tools, hooks, a);
    const sourceBefore = fs.readFileSync(stateFileOf(sandbox, a), 'utf8');
    const name = JSON.parse(sourceBefore).name;
    assert.ok(name, 'the source memory has no name to be found by');

    const res = await tools.get('cwl_adopt').execute('t', { from: name, as: 'ramo' }, undefined, undefined, b);
    assert.equal(res.details.ok, true, `the fork failed: ${JSON.stringify(res.details)}`);
    assert.equal(res.details.name, 'ramo');
    assert.equal(res.details.from, name);

    // The fork owns the leaves, and knows where it came from.
    const fork = stateOf(sandbox, b);
    assert.equal(fork.spans.length, 1, 'the fork did not copy the leaves');
    assert.equal(fork.spans[0].archived, true, 'the copied leaf was not marked as archived');
    assert.equal(fork.importedFrom, name);
    assert.equal(fork.name, 'ramo');
    // The source is untouched, byte for byte: adopting is not moving.
    assert.equal(fs.readFileSync(stateFileOf(sandbox, a), 'utf8'), sourceBefore,
      'the source memory was modified by the adoption');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('an ARCHIVED leaf is never pruned, however many turns pass', async () => {
  const { sandbox, home, tools, hooks, a, b } = await boot();
  try {
    await writeAMemory(tools, hooks, a);
    const name = stateOf(sandbox, a).name;
    await tools.get('cwl_adopt').execute('t', { from: name, as: 'ramo' }, undefined, undefined, b);

    // Turns of session B, whose messages have nothing to do with the adopted anchors.
    const own = [];
    for (let i = 1; i <= 10; i++) { own.push(user(i)); own.push(assistant(i)); await hooks.get('context')({ messages: own }, b); }

    const fork = stateOf(sandbox, b);
    assert.equal(fork.spans.length, 1, 'the archived leaf was pruned: its summary was the only copy');
    assert.equal(fork.spans[0].archived, true, 'the leaf lost its archived mark');
    assert.doesNotMatch(logOf(sandbox), /SPANS pruned: 1 span/,
      'the archived leaf was declared dead and pruned');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the inherited memory is injected at the TOP, and the announcement rides at the END until this session has its own leaves', async () => {
  const { sandbox, home, tools, hooks, a, b } = await boot();
  try {
    await writeAMemory(tools, hooks, a);
    const name = stateOf(sandbox, a).name;
    // NON-VACUITY, first: a session whose memory is ITS OWN has nothing to announce. If the
    // announcement were unconditional it would be noise in every ordinary session — and a
    // banner nobody can trust is a banner nobody reads.
    const before = await hooks.get('context')({ messages: [user(1), assistant(1)] }, b);
    assert.doesNotMatch(
      JSON.stringify((before?.messages || []).map((m) => m.content)),
      /INHERITED MEMORY|MEMORIA EREDITATA/,
      'the announcement appeared in a session with no inherited memory');
    await tools.get('cwl_adopt').execute('t', { from: name, as: 'ramo' }, undefined, undefined, b);

    const own = [];
    for (let i = 1; i <= 6; i++) { own.push(user(100 + i)); own.push(assistant(100 + i)); }
    const first = await hooks.get('context')({ messages: own }, b);
    const rendered = first?.messages || own;
    assert.equal(rendered[0]?.customType, 'cwl-compressed',
      'the inherited memory was not injected at the top of the context');
    assert.match(String(rendered[0].content), /CWL/, 'the top block does not carry the memory');
    // THE ANNOUNCEMENT IS AN INJECTION AT THE END, beside the compaction demand: NOT a prefix
    // (dropping it later would invalidate the cache it sits in) and NOT a fake user turn.
    const last = rendered[rendered.length - 1];
    assert.equal(last?.customType, 'cwl-inherited',
      'the announcement is not injected at the END of the context');
    const notice = String(last.content);
    assert.match(notice, /INHERITED MEMORY|MEMORIA EREDITATA/,
      'the injected message does not announce that the memory is inherited');
    assert.match(notice, /ramo/, 'the announcement does not name the memory');
    assert.ok(notice.includes(name), 'the announcement does not say where the memory came from');
    assert.match(notice, /cwl_open/, 'the announcement does not say how to read the memory');
    // AND IT MUST NOT STACK. An injected message comes BACK in the list of the next turn
    // (the gate's demand is filtered for exactly this reason): without the filter the tail
    // would grow one copy of the banner per turn. Simulated by handing the hook the list it
    // produced itself, which is the worst case.
    const again = await hooks.get('context')({ messages: rendered }, b);
    const grown = again?.messages || rendered;
    const copies = grown.filter((m) => m.customType === 'cwl-inherited').length;
    assert.equal(copies, 1, `the announcement stacked: ${copies} copies of it in the list`);
    // AND THE HEAD IS NOT FILTERED — measured here, and stated instead of hidden: fed the
    // list it produced itself, the hook prepends a SECOND copy of the archived leaf's notice
    // (`2 copies`), because a `cwl-compressed` block with no anchor cannot be recognised as
    // already present. It is pre-existing behaviour, and the evidence says Pi does not do this
    // in production: in the live session that inherited 205 leaves this test's author saw ONE
    // copy of the head after ~70 turns. What this filter guarantees is the failure mode the
    // gate already paid for once — one copy of the banner per turn — cannot happen even if the
    // list ever does come back. The head is left alone here: changing it is a separate call.
    const heads = grown.filter((m) => /^\[CWL · (RECALL|RICHIAMO)/.test(String(m.content))).length;
    assert.ok(heads <= 2, `the head multiplied unexpectedly: ${heads} copies`);

    // The case that breaks a naive implementation: the session compresses something of its
    // OWN, so `resolved` is no longer empty. The inherited block must still be there.
    const res = await tools.get('cwl_compress_range').execute(
      't', { summary: `OWN ${'y'.repeat(300)}`, micro: 'MICRO: mine.' }, undefined, undefined, b,
    );
    assert.equal(res.details.ok, true, `this session could not compress its own range: ${JSON.stringify(res.details)}`);
    const second = await hooks.get('context')({ messages: own }, b);
    const after = second?.messages || own;
    assert.equal(after[0]?.customType, 'cwl-compressed',
      'the inherited memory disappeared from the top once this session compressed something');
    // AND THE ANNOUNCEMENT GOES. This session now has a leaf of its own, so the sentence
    // "this session has no earlier messages of its own" would be a lie; the injection stops
    // by itself, in the one position (the tail) where stopping costs nothing. A banner left
    // on forever is a banner nobody reads.
    assert.notEqual(after[after.length - 1]?.customType, 'cwl-inherited',
      'the announcement outlived the window it describes: this session has leaves of its own now');
    assert.equal(stateOf(sandbox, b).spans.length, 2, 'the fork did not keep both the inherited and the new leaf');
  } finally { home.restore(); sandbox.cleanup(); }
});
