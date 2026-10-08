/**
 * THE MERGE REQUEST LEFT THE CONTEXT. THE CONDITION DID NOT.
 *
 * HISTORY, because the reversal matters and a test that loses its "why" is deleted by the
 * next person who finds it inconvenient.
 *
 * This test existed for a defect that was not a defect of the code but of the CHANNEL: the
 * merge of the old node worked and was covered by tests, and in a real session it could NOT
 * fire. `OLD NODE due: ...` was a LOG line: the extension knew it, the operator could read
 * it, and the only one who can write the summary — the agent — saw nothing. A mechanism
 * nobody can trigger is not a mechanism, so the request was moved into the CONTEXT.
 *
 * The operator has since taken the other decision, and it is the current contract: no request
 * that asks the agent to compress may enter the context. The merge is done by the BACKGROUND
 * PIT, which absorbs whole nodes with a synthesis of its own (`scheduleBackgroundPitSummary`
 * / `applyReadyBackgroundPitSummary`), and it needs no turn from anyone.
 *
 * So the three things this test now demands are:
 *  1. the request is NOT in the context — not when the merge is not due, not when it IS due;
 *  2. when it is due, the CONDITION is still visible where it must be: in the LOG, because the
 *     silent half of this story is exactly what cost a session;
 *  3. the merge still WORKS when something calls it (the background pit does, and so can the
 *     operator): the demand leaving the context must not quietly disable the mechanism.
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
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 2,
  // The size guards of a merge are OFF here: this test writes labels of a few characters,
  // and the real guard (3x a 3,600-character synthesis, and never less than 6,000) would
  // refuse the very merge the test is about. The guard has its own test in old-node.test.mjs.
  mergeMinRatio: 0,
  mergeMinChars: 0,
});

async function boot() {
  const sandbox = makeSandbox({ name: `trigger-${seq++}`, config: config() });
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
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

/** The request, if any, as the provider would see it. */
const demandRequest = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');

test('the merge request NEVER enters the context, and the condition stays in the log', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // 1. A single leaf: there is no node, so there is nothing to say.
    await hook(hooks, ctx, conversation(1, 4));
    const first = await tools.get('cwl_compress_range').execute('t', { summary: 'BODY-1' }, undefined, undefined, ctx);
    assert.equal(first.details.ok, true, `the first leaf was not born: ${JSON.stringify(first.details)}`);
    const withOne = await hook(hooks, ctx, conversation(1, 6));
    assert.equal(
      demandRequest(withOne).length,
      0,
      `the request is already there with ONE leaf and no node: a perpetual warning is noise, and noise teaches to ignore warnings. Found: ${demandRequest(withOne).slice(0, 160)}`,
    );

    // 2. Five leaves with a micro: two young nodes form, and the merge becomes due.
    for (let i = 2; i <= 5; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const leaves = stateOf(sandbox).spans;
    for (let i = 0; i < leaves.length; i++) {
      await tools.get('cwl_micro').execute('t', { id: leaves[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
    }

    const logBefore = logOf(sandbox).length;
    const due = await hook(hooks, ctx, conversation(1, 12));
    const text = demandRequest(due);
    assert.equal(
      text.length,
      0,
      `the merge is due and the request must still NOT be in the context: the operator removed every request that `
        + `asks the agent to compress, and the merge is done by the background pit. Found: ${text.slice(0, 200)}`,
    );

    // 2b. The CONDITION is due and it is SAID, in the log. This is the half of the story that
    // must not go silent: the demand left the context, the diagnosis did not.
    const rows = logOf(sandbox).slice(logBefore);
    assert.match(
      rows,
      /OLD NODE due: 2\+ young nodes/,
      `the merge being due must still be stated in the LOG, with the number of nodes it starts at: ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      rows,
      /LOG ONLY now/,
      `the log row must say that the log is the only channel left, so nobody reads it as "the agent was asked": ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );

    // 2c. And the state agrees: two young nodes really are waiting for the merge.
    const young = stateOf(sandbox).nodes.filter((nd) => nd.id !== (stateOf(sandbox).oldNode?.id ?? ''));
    assert.ok(
      young.length >= 2,
      `the fixture must still put the merge in a state where it is DUE, or this test proves nothing: ${JSON.stringify(stateOf(sandbox).nodes)}`,
    );

    // 3. Merged: the merge is no longer due, and no request ever appeared. The mechanism
    // still WORKS — it is the demand that left, not the merge.
    const merged = await tools.get('cwl_old').execute('t', { text: 'BIG-SUMMARY-1' }, undefined, undefined, ctx);
    assert.equal(merged.details.ok, true, `the merge failed: ${JSON.stringify(merged.details)}`);
    const after = await hook(hooks, ctx, conversation(1, 12));
    assert.equal(
      demandRequest(after).length,
      0,
      `no request may ever appear: ${demandRequest(after).slice(0, 200)}`,
    );
  } finally {
    home.restore();
  }
});
