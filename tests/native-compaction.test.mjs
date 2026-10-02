/**
 * Level A: the encounter with the native compaction of Pi.
 *
 * Why it exists. The native compaction of Pi cuts a PREFIX: it keeps
 * `firstKeptEntryId` and everything that comes after it, and prepends its own
 * summary. Two consequences, both verified here.
 *
 * 1. An episode whose OPENING anchor the cut carried away (with implicit opens that
 *    anchor is the PREVIOUS episode's closing) keeps its own closing one. Its surviving
 *    part is everything the list still holds up to that closing, so the range must be
 *    DEDUCED ([0, end]) instead of read.
 *
 * 2. That range starts at index 0, where the summary of the native
 *    compaction lives: a message with role 'compactionSummary'
 *    (pi/dist/core/messages.js, createCompactionSummaryMessage) which is NOT a
 *    user turn and which nobody protected. Evacuating it destroys the only copy
 *    of the history that the compaction replaced: the transcript keeps it, the
 *    provider does not.
 *
 * The opposite direction (closing lost, opening alive -> [start, len-1]) is
 * REJECTED, and there is a test that keeps it rejected: a prefix cut cannot
 * take away the closing while leaving the opening, so that arrangement
 * has no explanation, and inventing a range for an unexplained arrangement is
 * exactly the way this extension would evacuate what it does not know how to
 * count.
 *
 * MEASURED before writing the branch (189 transcripts, 202 native compactions,
 * 7 closed episodes): 5 episodes stood entirely BEFORE the cut — their
 * TWO anchors had vanished, and no deduction can save them — 2 entirely
 * after, 0 straddling. The branch therefore repairs nothing that was observed:
 * it is an insurance for the arrangement that the cut makes possible. That is
 * why the test also demands that the LOG says it happened.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Full evacuation only: it isolates the deduction from the three strip levels. With the
 * strip levels off, the escalation of the plan has no other choice than
 * reaching `removed`, which is the level where the range really matters.
 */
const onlyRemoval = (extra = {}) => ({
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
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => {
  const p = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
};

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const text = (m) => JSON.stringify(m ?? {});
const contains = (out, marker) => out.some((m) => text(m).includes(marker));
/** Filler big enough to make the `removed` level necessary. */
const assistant = (marker) => ({
  role: 'assistant',
  content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }],
});

const SUMMARY = 'SUMMARY BORN FROM THE CUT ' + 'S'.repeat(4000);
const summaryMessage = () => ({
  role: 'compactionSummary',
  summary: SUMMARY,
  tokensBefore: 90000,
  timestamp: 1,
});

/**
 * Records ONE episode. With implicit opens its start anchor is the PREVIOUS episode's
 * closing anchor, so a chain of calls is what produces a chain of episodes.
 */
async function recordEpisode(tools, ctx, endAnchor, name, type = 'expl', description = 'learned') {
  await tools.get('delimiter').execute(endAnchor, { name, type, description }, undefined, undefined, ctx);
}

test('an episode whose opening was lost is located by deduction and evacuated', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval());
  try {
    // A chain of two: the SECOND episode begins where the FIRST ended, and that
    // closing anchor is the one the cut carries away while the second end survives.
    await recordEpisode(tools, ctx, 'call-gone', 'first');
    await recordEpisode(tools, ctx, 'call-e', 'crosses');
    const messages = [
      summaryMessage(),
      assistant('INSIDE-1'),
      assistant('INSIDE-2'),
      // The closing anchor, alive: it is the only endpoint left. It lives at index 3,
      // so the deduced range is [0, 3].
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(!contains(out, 'INSIDE-1') && !contains(out, 'INSIDE-2'),
      'the range was not deduced: the episode stayed invisible to the evacuation');
    // The log must say it. A range deduced instead of read is a thing that
    // the operator must be able to see, otherwise the deduction is an assumption.
    assert.match(logOf(sandbox), /EPISODES deduced: 1/, 'the log did not say that a range was deduced');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the summary of the native compaction survives that range', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval());
  try {
    await recordEpisode(tools, ctx, 'call-gone', 'first');
    await recordEpisode(tools, ctx, 'call-e', 'crosses');
    const messages = [
      summaryMessage(),
      assistant('INSIDE-1'),
      assistant('INSIDE-2'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    // The deduced range was applied (otherwise this test would prove
    // nothing about the summary: it would be alive only because nobody touched it).
    assert.ok(!contains(out, 'INSIDE-1'), 'the range was not applied: the test proves nothing');
    assert.ok(contains(out, SUMMARY),
      'the summary of the native compaction was evacuated: it was the only copy of the replaced history');
    assert.ok(out.some((m) => m.role === 'user'), 'a user turn was touched');
    assert.match(logOf(sandbox), /SUMMARY GUARD: 1/, 'the guard did not count the summary it saved');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the opposite direction is NOT deduced: no range invented for an episode whose closing is gone', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(onlyRemoval());
  try {
    // The FIRST episode closes on an anchor still in the list; the SECOND one's own
    // closing anchor is gone. A prefix cut cannot produce that — the closing comes AFTER
    // the start — so the arrangement has no explanation and no range is invented for it.
    await recordEpisode(tools, ctx, 'call-s', 'first');
    await recordEpisode(tools, ctx, 'call-gone', 'without-closing');
    const messages = [
      { role: 'user', content: 'opening' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'open' }] },
      assistant('INTACT-1'),
      assistant('INTACT-2'),
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(contains(out, 'INTACT-1') && contains(out, 'INTACT-2'),
      'a range [start, len-1] was invented for an episode whose closing is unexplained');
    assert.ok(!/EPISODES deduced: [1-9]/.test(logOf(sandbox)), 'the log declares a deduction that must not exist');
  } finally { home.restore(); sandbox.cleanup(); }
});

// The old test "a located episode wins over the deduction inside its own range" is
// GONE, and deliberately: with IMPLICIT opens the episodes form a CHAIN — an episode
// begins exactly where the previous one ended — so a deduced range can never reach
// past its own closing anchor into the content of the episode that follows it. That
// overlap is structurally impossible now, and a test for an impossible arrangement
// would be a lie. The equivalent guard for SPANS, which CAN overlap, lives in
// tests/span-vs-eviction.test.mjs.
