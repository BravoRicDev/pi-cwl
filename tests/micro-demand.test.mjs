/**
 * THE MICRO NO LONGER ASKS: IT IS SAID IN THE LOG, AND WRITTEN IN THE BACKGROUND.
 *
 * HISTORY, because the reversal matters and a test that loses its "why" gets deleted by the
 * next person who finds it inconvenient.
 *
 * MEASURED LIVE, right after the reload that let the index run:
 *
 *     NODES: 0 node(s) [none], 40 leaf/leaves waiting for a micro — their body is still in
 *     the context
 *
 * on EVERY turn, with 45 spans and 90.415t of summaries still in the context. The code ran,
 * but the mechanism could not start: leaves without a micro do not enter a node -> zero
 * nodes -> `plan.due` is zero -> the merge request (`indexDue`) NEVER appeared. And nobody
 * talked about the micro: the count ended up in the LOG, not in the context.
 *
 * The answer THEN was to put the request in the context (`bd0734c` closed it for `cwl_old`,
 * this covers the step BEFORE). The operator has since taken the other decision, and it is
 * the current contract: NO request that asks the agent to compress may enter the context.
 * The micro is written by the BACKGROUND SUMMARIZER, which produces the micro together with
 * the summary — that is why a leaf can be born ready for a node without a turn from anyone.
 *
 * So what this test pins is no longer the presence of a message. It pins three things:
 *  1. the ABSENCE of the demand, in the very scenario that used to produce it;
 *  2. that the CONDITION still exists in the state (leaves without a micro), so the test
 *     cannot pass by accident on an empty fixture;
 *  3. that the condition is still SAID OUT LOUD in the log. A mechanism that stops talking
 *     silently is a mechanism nobody can diagnose, and this one already cost a session once.
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
  const sandbox = makeSandbox({ name: `micro-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const stateOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

// Not exported by `_helpers.mjs`: every test that reads the log defines it in house.
const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('the micro request NEVER enters the context: the leaves wait in silence, and the log says so', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // SEVEN leaves, not one. With `looseLeaves` = 5 the last five stay LOOSE and do not
    // owe a micro yet: the request is born only when a leaf leaves that
    // window, and that is exactly the case that matters (its body must leave the
    // context). My first fixture made ONE and the test was red for that reason, not for
    // the code: the usual lesson, the defect was in the test.
    for (let i = 1; i <= 7; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    assert.equal(
      stateOf(sandbox).spans.length,
      7,
      'seven leaves are needed: with five or fewer they are all loose and none is waiting for a micro',
    );

    // The turn after: the older leaves are no longer loose, they have no micro, and the
    // extension knows it. The agent is NOT told, by decision.
    const out = await hook(hooks, ctx, conversation(1, 10));
    const requests = out.filter((m) => m.customType === 'cwl-demand');
    assert.equal(
      requests.length,
      0,
      `no demand may enter the context any more: the operator removed every request that asks the agent to compress, `
        + `and the micro is written by the background summarizer. Messages: ${out.map((m) => m.customType || m.role).join(', ')}`,
    );

    // (2) The CONDITION is still there. Without this the test would pass on any fixture at
    // all, including one that never produces a leaf without a micro.
    const waiting = stateOf(sandbox).spans.filter((sp) => !String(sp.micro ?? '').trim());
    assert.equal(
      waiting.length,
      7,
      `the fixture must still produce leaves WITHOUT a micro, or this test proves nothing: ${JSON.stringify(
        stateOf(sandbox).spans.map((sp) => ({ id: sp.id, micro: sp.micro ?? null })),
      )}`,
    );

    // (3) And it is still said out loud, in the log. Nothing is lost by the demand leaving
    // the context as long as this row is here: the operator can still see it.
    const log = logOf(sandbox);
    assert.match(
      log,
      /MICRO due: \d+ leaf\/leaves waiting for a micro/,
      `the condition must still be stated in the LOG — a mechanism that goes silent cannot be diagnosed: ${log.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      log,
      /LOG ONLY now/,
      `the log row must say that it is the only channel left, so nobody reads it as "the agent was asked": ${log.trim().split('\n').slice(-4).join(' | ')}`,
    );
  } finally {
    home.restore();
  }
});
