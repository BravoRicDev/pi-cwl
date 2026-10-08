/**
 * THE WINDOW THE INDEX OPENS MUST BE NAMED WHILE IT IS STILL OPEN.
 *
 * A topic can only be born from the leaves of the BUFFER — the last node, the one attached
 * to the leaves still open. `cwl_group` refuses anything else (`leaf-not-in-the-buffer`),
 * because a leaf that has entered a node can never be moved again. So the index CLOSES the
 * window that the topic needs, and a topic that is not born while its leaves are still in
 * the buffer is lost for good.
 *
 * Until now the agent learned the size guard the hard way: by trying, being refused, and
 * paying a turn and a failed call. MEASURED: the guard is
 * `mergeMinRatio x MERGE_SYNTHESIS_CHARS`, never less than `mergeMinChars` — 10,800
 * characters with the defaults. The invitation says it BEFORE, with every number needed to
 * act: which node, how many leaves, how many characters of micro they hold, and how many
 * the guard wants.
 *
 * The invitation is deliberately NOT unconditional: it fires only when the guard would
 * PASS. An invitation that cannot be acted on teaches nothing and costs a demand per turn.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * `looseLeaves: 1` — NOT zero. `leaves.slice(-0)` is `leaves.slice(0)`: zero loose leaves
 * would make EVERY leaf loose and no node would ever be born. One keeps the newest leaf
 * out and lets the rest enter the buffer, so `n` filled leaves give `n - 1` in the node.
 * `mergeMinRatio: 0` takes the ratio term out of `needChars`, so the test drives the floor
 * alone and the fixture stays small.
 */
const config = (over = {}) => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  looseLeaves: 1,
  nodeCapacity: 30,
  mergeNodesAt: 3,
  mergeMinRatio: 0,
  mergeMinChars: 100,
  gate: false,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...over,
});

async function boot(over) {
  const sandbox = makeSandbox({ name: `topic-${seq++}`, config: config(over) });
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

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `round ${i} paragraph ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `response ${i} paragraph ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

/** `n` leaves, each born WITH a micro of exactly `microChars`, so all of them can enter a node. */
async function fillBuffer(hooks, ctx, tools, n, microChars) {
  for (let i = 1; i <= n; i++) {
    await hook(hooks, ctx, conversation(1, i + 3));
    const res = await tools.get('cwl_compress_range').execute(
      't',
      { summary: 'x'.repeat(300), micro: 'm'.repeat(microChars) },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
  }
}

const demandText = (messages) =>
  messages
    .filter((m) => m.customType === 'cwl-demand')
    .map((m) => String(m.content))
    .join('\n');

// Not exported by `_helpers.mjs`: every test that reads the log defines it in house.
const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

test('past 18 leaves the buffer meets the topic condition, and it is said in the log only', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await fillBuffer(hooks, ctx, tools, 20, 20);
    // A leaf enters a node on the hook AFTER the one that created it: the state read here
    // is one turn behind, and the invitation is computed on the hook below.
    const out = await hook(hooks, ctx, conversation(1, 30));
    const st = stateOf(sandbox);
    assert.equal(st.spans.length, 20, 'twenty leaves were expected');

    const buffer = st.nodes[st.nodes.length - 1];
    assert.ok(buffer, 'the buffer node was not created');
    assert.equal(buffer.leaves.length, 19, 'nineteen leaves must be in the buffer: the twentieth is loose');
    // The premise of the whole mechanism: the leaves are STILL movable.
    assert.equal(
      st.nodes.length,
      1,
      'the buffer must be the last node and the only one, otherwise the leaves are not movable',
    );

    // THE INVITATION LEFT THE CONTEXT. The CONDITION did not, and neither did its numbers:
    // they are in the LOG, which is now the only channel. The state assertions above are what
    // keep this test honest — the window the topic needs really is open, and the leaves are
    // still movable.
    const text = demandText(out);
    assert.equal(
      text.length,
      0,
      `no topic invitation may enter the context any more: the operator removed every request `
        + `that asks the agent to compress. Messages: ${out.map((m) => m.customType || m.role).join(', ')}`,
    );

    const rows = logOf(sandbox);
    assert.ok(
      rows.includes(`TOPIC due: the buffer ${buffer.id} holds 19 leaf/leaves`),
      `the topic condition must still be stated in the LOG, naming the buffer and its leaf count: ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      rows,
      /TOPIC due:.*1900 characters|TOPIC due:.*380 chars/,
      `the log row must carry how many characters of micro the leaves hold (19 x 20 = 380): ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      rows,
      /need 100/,
      `the log row must carry how many characters the guard wants (mergeMinChars = 100), or nobody can see whether `
        + `the group would pass: ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );
    assert.match(
      rows,
      /LOG ONLY now/,
      `the log row must say that the log is the only channel left: ${rows.trim().split('\n').slice(-4).join(' | ')}`,
    );
  } finally {
    home.restore();
  }
});

test('exactly 18 leaves: the invitation has NOT fired yet', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await fillBuffer(hooks, ctx, tools, 19, 20);
    await hook(hooks, ctx, conversation(1, 30));
    assert.equal(stateOf(sandbox).nodes[0].leaves.length, 18, 'eighteen leaves were expected in the buffer');
    // The demand can no longer be the probe: it never appears. The LOG is the probe now, and
    // it is a STRICTER one — it would catch a condition computed anyway and merely not sent.
    assert.ok(
      !logOf(sandbox).includes('TOPIC due'),
      'the topic condition fired at 18 leaves, one turn early: the threshold is "past 18", so 19',
    );
    assert.equal(
      demandText(await hook(hooks, ctx, conversation(1, 30))).length,
      0,
      'no demand may ever enter the context',
    );
  } finally {
    home.restore();
  }
});

test('nineteen leaves but too few characters: no invitation, the guard would refuse', async () => {
  // The floor is raised above what the nineteen leaves hold (19 x 20 = 380), so the group
  // would be refused by `groupTooSmall`. Inviting anyway would send the agent into a wall.
  const { sandbox, home, tools, hooks, ctx } = await boot({ mergeMinChars: 100_000 });
  try {
    await fillBuffer(hooks, ctx, tools, 20, 20);
    await hook(hooks, ctx, conversation(1, 30));
    assert.equal(stateOf(sandbox).nodes[0].leaves.length, 19, 'nineteen leaves were expected in the buffer');
    assert.ok(
      !logOf(sandbox).includes('TOPIC due'),
      'the topic condition fired although the group would be refused (380 chars of micro against a floor of '
        + '100,000): a condition nobody can act on teaches nothing',
    );
  } finally {
    home.restore();
  }
});
