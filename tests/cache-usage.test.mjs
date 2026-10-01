/**
 * THE CACHE ROW: THE INSTRUMENT THAT CHARGES THE COST TO THE EVENT.
 *
 * Why it exists. The provider cache is a PREFIX cache: everything that changes
 * from one turn to the next has to stay at the END, and the head must stay
 * immutable. Every leaf written by CWL lands where the compressed content used to
 * be, and a merge lands at the very FRONT — both invalidate everything after them,
 * and the next request pays a WRITE where it used to pay a READ.
 *
 * The row `CACHE read=… write=… input=… output=… hit=…% after=<event>
 * since-compress=<n>` is what makes that cost VISIBLE, and it is the only way to
 * tell whether batching the writes was worth anything. `after=` is the whole point:
 * without it the numbers exist but cannot be attributed, and an unattributed cost
 * cannot be reduced.
 *
 * Two properties are load-bearing and neither is visible from a green run of the
 * extension itself:
 *  1. ONE row per REQUEST, not per hook call: a tool loop calls the context hook
 *     many times for the same request, and the row would be charged many times;
 *  2. the event is SPENT by the row that shows it: the next row must not repeat
 *     `after=new-leaf` for a write it did not cause.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 100_000,
  thresholdRatio: 1,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(extra = {}) {
  const sandbox = makeSandbox({ name: `cache-${seq++}`, config: config(extra) });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

/** The rows, verbatim: the test compares them whole, not field by field. */
const cacheRows = (sandbox) =>
  [...logOf(sandbox).matchAll(/CACHE read=\d+ write=\d+ input=\d+ output=\d+ hit=\d+% after=[a-z-]+ since-compress=-?\d+/g)].map((m) => m[0]);

const user = (i) => ({ role: 'user', content: `prompt ${i} ${'P'.repeat(200)}` });
const assistant = (i) => ({ role: 'assistant', content: `answer ${i} ${'A'.repeat(200)}` });
/** An assistant turn as Pi hands it over: `usage` is NOT in the public union. */
const withUsage = (ts, usage) => ({ role: 'assistant', content: 'answer body', timestamp: ts, usage });

test('the cache row is written once per request, with the numbers it read', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    const messages = [
      user(1),
      withUsage(111, { cacheRead: 900, cacheWrite: 100, input: 0, output: 50 }),
    ];
    // The hook is free to answer NOTHING when it changes nothing (under the trigger,
    // with no event to apply): what this test watches is the log, not the answer.
    await hooks.get('context')({ messages }, ctx);
    const rows = cacheRows(sandbox);
    assert.equal(rows.length, 1, `the row was not written: ${logOf(sandbox).slice(-400)}`);
    // hit = cacheRead / (cacheRead + cacheWrite + input) = 900 / 1000.
    assert.equal(rows[0], 'CACHE read=900 write=100 input=0 output=50 hit=90% after=none since-compress=-1');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the same usage timestamp is not charged twice, a new one is', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    const messages = [
      user(1),
      withUsage(111, { cacheRead: 900, cacheWrite: 100, input: 0, output: 50 }),
    ];
    // The SAME request calls this hook more than once (a tool loop): the row must
    // stay one. Before the dedupe on the timestamp it was charged once per call.
    await hooks.get('context')({ messages }, ctx);
    await hooks.get('context')({ messages }, ctx);
    assert.equal(cacheRows(sandbox).length, 1,
      'the same usage timestamp was charged twice: one request, one row');

    // A NEW request carries new numbers, and it must be visible.
    await hooks.get('context')({
      messages: [...messages, withUsage(222, { cacheRead: 700, cacheWrite: 300, input: 0, output: 10 })],
    }, ctx);
    const rows = cacheRows(sandbox);
    assert.equal(rows.length, 2, 'a new usage must be a new row');
    assert.equal(rows[1], 'CACHE read=700 write=300 input=0 output=10 hit=70% after=none since-compress=-1');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the row charges a leaf to the event that wrote it, and that event is spent', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({ tokenBudget: 300, thresholdRatio: 0.5 });
  try {
    const messages = [];
    for (let i = 1; i <= 6; i++) { messages.push(user(i)); messages.push(assistant(i)); }
    await hooks.get('context')({ messages }, ctx);

    const res = await tools.get('cwl_compress_range').execute(
      't', { summary: `BODY ${'x'.repeat(300)}`, micro: 'MICRO: the points that matter most.' }, undefined, undefined, ctx,
    );
    assert.equal(res.details.ok, true, `the leaf was not born: ${JSON.stringify(res.details)}`);

    // The NEXT request carries new numbers: the write it pays for was caused by the
    // leaf just written, and the row must say so — `after=new-leaf`, 0 turns since.
    await hooks.get('context')({
      messages: [...messages, withUsage(5000, { cacheRead: 100, cacheWrite: 900, input: 0, output: 0 })],
    }, ctx);
    let rows = cacheRows(sandbox);
    assert.match(rows[rows.length - 1], /after=new-leaf since-compress=0$/,
      `the row did not charge the leaf: ${rows[rows.length - 1]}`);

    // The event explains ONE row: the next one is `none` again, or a stale cause
    // would be re-charged to an event that happened many turns earlier.
    await hooks.get('context')({
      messages: [...messages, withUsage(6000, { cacheRead: 100, cacheWrite: 900, input: 0, output: 0 })],
    }, ctx);
    rows = cacheRows(sandbox);
    assert.match(rows[rows.length - 1], /after=none since-compress=/,
      `the event was charged twice: ${rows[rows.length - 1]}`);
  } finally { home.restore(); sandbox.cleanup(); }
});
