/**
 * Eviction accounting: the log line must not declare two
 * different savings.
 *
 * THE DEFECT, MEASURED. The line
 *   `EVICTION applied: 32 msg removed, 0 reduced, 79683t -> 58987t (saved 21053t)`
 * declares 21053t but the real eviction is `79683 - 58987 = 20696t`. The number
 * in parentheses was the SUM OF THE PLAN ESTIMATES (`removedTokens +
 * truncatedTokens`), not the saving: two different quantities printed as if
 * they were the same. Other lines of the same log were wrong in the other direction
 * (77164 -> 68800 = 8364, it declared 8056), and it is precisely the alternation of the signs
 * that proves it is not a rounding.
 *
 * THE THREE CAUSES, all inside the applier's body:
 *  1. the episode's re-entry MARKER (`role:'custom'`, `cwl-evicted`)
 *     is ADDED to the list: it was not in the input, so it raises `afterTokens`
 *     and the real saving is SMALLER than the declared one (+357t on that line);
 *  2. pruning the orphan `toolCall`s SHRINKS surviving messages,
 *     and nobody counts it: the real saving is LARGER (-308t, -121t);
 *  3. for the strips `truncatedTokens` measures `textLengthOf` (the content only)
 *     while `currentTokens`/`afterTokens` measure the SERIALIZED message
 *     (`JSON.stringify`): two different bases in the same line.
 *
 * And the message the user reads (`evictionNotice`, which receives
 * `currentTokens` and `afterTokens`) ALREADY used the measured number: log and
 * interface told two different things about the same event, and the state
 * counter (`totalEvictedTokens`) followed the estimate.
 *
 * The test does not demand that the plan be precise: it demands that the line be
 * ARITHMETICALLY ONE. An operator who reads two numbers and finds them different no
 * longer knows which to believe, and that is exactly how this defect was
 * found.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/** Full eviction only: the `removed` level is the one that adds the marker. */
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
  const sandbox = makeSandbox({ name: `accounting-${seq++}`, config });
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

const assistant = (marker) => ({
  role: 'assistant',
  content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }],
});

/** The line I am measuring, read from the real log. */
const RIGA = /EVICTION applied: (\d+) msg removed, (\d+) reduced, (\d+)t -> (\d+)t \(saved (\d+)t\)/;

test('the EVICTION line declares the MEASURED saving, not the plan estimate', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    // A closed episode entirely inside the range: its full eviction
    // is what adds the re-entry marker (cause no.1 of the defect).
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'accountancy', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'accountancy', description: 'takeaway' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: 'prologue' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      assistant('INSIDE-1'),
      assistant('INSIDE-2'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(!JSON.stringify(out).includes('INSIDE-1'),
      'no eviction happened: the test proves nothing');

    const m = RIGA.exec(logDi(sandbox));
    assert.ok(m, 'the EVICTION line was not written to the log');
    const [, dropped, , before, after, saved] = m.map(Number);
    assert.ok(dropped >= 1, 'the line says zero messages removed: this is not an eviction');
    assert.equal(
      saved,
      before - after,
      `the line declares ${saved}t saved but ${before}t -> ${after}t makes ${before - after}: ` +
      'two different numbers in the same line, and in between there are the added marker, the pruned orphan toolCalls ' +
      'and two estimators (content against serialized JSON)',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the EVICTION line is consistent also when the eviction REDUCES instead of removing', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({
    tokenBudget: 1000,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: true, stripIntermediate: false, removeEpisode: false },
    showWidget: false,
    debug: true,
  });
  try {
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'reduction', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'reduction', description: 'takeaway' }, undefined, undefined, ctx);
    // A huge toolResult inside the episode: the `bulk` level cuts it without
    // removing it, and it is the branch where `truncatedTokens` uses the other estimator.
    const grasso = { role: 'toolResult', toolCallId: 'call-big', toolName: 'bash', content: [{ type: 'text', text: 'B'.repeat(60000) }] };
    const messages = [
      { role: 'user', content: 'prologue' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'opened' }] },
      grasso,
      assistant('INSIDE-1'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'closed' }] },
      { role: 'user', content: 'recent question' },
    ];
    const out = await hook(hooks, ctx, messages);
    const ridotto = JSON.stringify(out).length < JSON.stringify(messages).length;
    assert.ok(ridotto, 'no reduction happened: the test proves nothing');

    const m = RIGA.exec(logDi(sandbox));
    assert.ok(m, 'the EVICTION line was not written to the log');
    const [, , , before, after, saved] = m.map(Number);
    assert.equal(
      saved,
      before - after,
      `the line declares ${saved}t saved but ${before}t -> ${after}t makes ${before - after}: ` +
      'for the strips the count uses the length of the CONTENT while the arrow uses the SERIALIZED message',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});
