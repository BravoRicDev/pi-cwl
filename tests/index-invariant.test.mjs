/**
 * THE INDEX INVARIANT — the first test of the plan (`PLAN-SUMMARY-INDEX.md`, sec. 3).
 *
 * The project is about to build an index of summaries (leaves -> node of 30 -> old
 * node) that must stay CONTIGUOUS and NON-OVERLAPPING: no hole (nothing lost) and
 * no overlap (nothing counted twice). Before writing that code, the invariant
 * must be tried on what already exists: the spans.
 *
 * TWO HOLES FOUND BY READING THE CODE, and this test pins them down.
 *
 * 1. `locateSpans` (index.ts 1772-1831) SILENTLY discards the span contained in
 *    another:
 *
 *      .filter((x, _all, arr) => !arr.some((o) => o !== x && o.from <= x.from && o.to >= x.to))
 *
 *    The `dead` set is filled earlier, in the `map`, so a contained span ends up
 *    neither in `resolved` nor in `dead`: it disappears from every count. And the
 *    log says `SPANS applied: ${applied.applied}`, that is, `resolved.length`, so
 *    there is no line declaring the disappearance. Its summary stays in the
 *    state and is re-saved on every turn: dead weight nobody counts.
 *
 * 2. There is no check at all on PARTIAL overlaps. The filter above
 *    covers only containment: two spans that intersect without containing each
 *    other survive BOTH, and the shared region is counted twice (also
 *    by `declareFloor`, which adds up `spanTokens`).
 *
 * It is the same family as the defects already closed in this session (`saved` that
 * was the plan's estimate, the inflated span saving): a number claiming more
 * than the work done, or something disappearing without being declared.
 *
 * The test builds the two spans WITH THE AGENT'S TOOLS, not by hand: the question
 * is not whether `locateSpans` can handle a theoretical case, but whether the agent
 * can produce it. `cwl_compress` (index.ts ~2159-2197) only checks that the three
 * parameters are there, that the start+end pair does not already exist (then it
 * revokes) and that both hashes are in `knownHashes`: no guard on already covered
 * regions.
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
  const sandbox = makeSandbox({ name: `invariant-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

/** The session state: the sandbox holds only one, so the key hash is not needed. */
const stateOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const compress = (tools, ctx, startHash, endHash, summary) =>
  tools.get('cwl_compress').execute('t', { startHash, endHash, summary }, undefined, undefined, ctx);

/** The path by which the agent compresses: the extension picks it, and the tool SAVES the state. */
const compressRange = (tools, ctx, summary) =>
  tools.get('cwl_compress_range').execute('t', { summary }, undefined, undefined, ctx);

/** `user`/`assistant` pairs: the user turns survive the spans, the assistants do not. */
const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

test('a span contained in another cannot disappear without being declared', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Turn 1: the extension takes the measurements and knows which region to compress.
    await hook(hooks, ctx, conversation(1, 6));

    // Span A: the first, on the region the extension chose. The tool saves the
    // state, so from here on the state is readable on disk.
    const a = await compressRange(tools, ctx, 'SUMMARY-A');
    assert.equal(a.details.ok, true, `span A not created: ${JSON.stringify(a.details)}`);
    await hook(hooks, ctx, conversation(1, 6));

    // The conversation GROWS, so that a second compressible region exists: without
    // it there would be nothing with which to contain A.
    await hook(hooks, ctx, conversation(1, 9));
    const b = await compressRange(tools, ctx, 'SUMMARY-B');
    assert.equal(b.details.ok, true, `span B not created: ${JSON.stringify(b.details)}`);

    const afterB = stateOf(sandbox).spans;
    assert.equal(afterB.length, 2, `TWO distinct spans are needed for the invariant to have an object: ${JSON.stringify(afterB)}`);
    const [spanA, spanB] = afterB;

    // Span OUTER: from A.start to B.end. It CONTAINS both A and B.
    const outer = await compress(tools, ctx, spanA.startHash, spanB.endHash, 'SUMMARY-OUTER');
    assert.equal(outer.details.ok, true, `span OUTER not created: ${JSON.stringify(outer.details)}`);
    assert.equal(stateOf(sandbox).spans.length, 3, 'the third span did not make it into the state: the test\'s premise does not hold');

    // Final turn: `locateSpans` sees A and B inside OUTER. Only what
    // this turn wrote is looked at — the log accumulates previous turns, and reading
    // the FIRST line instead of the last makes the test pass for the wrong reason.
    // It really happened: the mutation that applied AND pruned the same spans
    // (resolved = all, dead = the contained ones) passed, because the first applied
    // line of the log was the one from the turn in which A was born.
    const beforeTurn = logOf(sandbox).length;
    await hook(hooks, ctx, conversation(1, 9));
    const log = logOf(sandbox).slice(beforeTurn);
    const applied = Number((/SPANS applied: (\d+)/.exec(log) || [0, 0])[1]);
    const pruned = Number((/SPANS pruned: (\d+)/.exec(log) || [0, 0])[1]);

    assert.equal(
      applied + pruned,
      3,
      `three spans in the state, but the log declares ${applied} applied and ${pruned} pruned: the others vanished silently. ` +
        'The containment filter of locateSpans discards them without recording them in `dead` (index.ts 1772-1831), ' +
        'so they are neither applied nor declared: their summaries stay in the state weighing on every turn.',
    );
  } finally {
    home.restore();
  }
});

test('two spans that overlap without containing each other are DECLARED', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await hook(hooks, ctx, conversation(1, 6));
    const a = await compressRange(tools, ctx, 'SUMMARY-A');
    assert.equal(a.details.ok, true, `span A not created: ${JSON.stringify(a.details)}`);
    await hook(hooks, ctx, conversation(1, 6));

    await hook(hooks, ctx, conversation(1, 9));
    const b = await compressRange(tools, ctx, 'SUMMARY-B');
    assert.equal(b.details.ok, true, `span B not created: ${JSON.stringify(b.details)}`);
    const [spanA, spanB] = stateOf(sandbox).spans;
    assert.ok(spanA && spanB, 'TWO distinct spans are needed to build the overlap');

    // OVER starts from the END of A: it shares with A the last address (a user
    // turn, which survives) and contains B. The containment filter has
    // nothing to say: A and OVER do not contain each other.
    const over = await compress(tools, ctx, spanA.endHash, spanB.endHash, 'SUMMARY-OVER');
    assert.equal(over.details.ok, true, `span OVER not created: ${JSON.stringify(over.details)}`);
    assert.equal(stateOf(sandbox).spans.length, 3, 'the third span did not make it into the state');

    const beforeTurn = logOf(sandbox).length;
    await hook(hooks, ctx, conversation(1, 9));
    const log = logOf(sandbox).slice(beforeTurn);

    // Nothing is lost: A and OVER stay applied (discarding A would bring back into
    // the context the messages only A covers), B is contained in OVER and gets pruned.
    const applied = Number((/SPANS applied: (\d+)/.exec(log) || [0, 0])[1]);
    const pruned = Number((/SPANS pruned: (\d+)/.exec(log) || [0, 0])[1]);
    assert.equal(applied + pruned, 3, `nothing can vanish: applied ${applied} + pruned ${pruned} instead of 3`);

    // But the shared region is described by TWO summaries: it must be said, not undergone.
    const overlapMatch = /SPANS overlap: (\d+)/.exec(log);
    assert.ok(
      overlapMatch && Number(overlapMatch[1]) === 1,
      `a pair of spans shares a region and no line declares it: A=[0..A.end] and OVER=[A.end..B.end] ` +
        'touch at one index, so that message is inside two summaries. Today `locateSpans` has no ' +
        `check on partial overlaps and no count sees them. Turn log: ${log.trim().split('\n').slice(-4).join(' | ')}`,
    );
  } finally {
    home.restore();
  }
});
