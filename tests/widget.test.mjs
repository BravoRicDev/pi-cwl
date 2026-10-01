/**
 * THE TUI WIDGET — the index shape in the window, and NOT in the conversation.
 *
 * `showWidget` was declared, given a default of `true`, validated by the config reader
 * and read by NOBODY: there was no `ctx.ui.setWidget` anywhere in the extension. An
 * option that promises something and draws nothing.
 *
 * The point of a widget here is that it is UI: it costs no token, it cannot nudge the
 * agent, and it must never turn into a message. These two tests hold both ends:
 *  1. the line is drawn, and it carries the shape of the index;
 *  2. nothing that was drawn reaches the messages of the turn.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: true,
  debug: true,
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 2,
  mergeMinRatio: 0,
  mergeMinChars: 0,
  ...extra,
});

async function boot(extra) {
  const sandbox = makeSandbox({ name: `widget-${seq++}`, config: config(extra) });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const drawn = [];
  const ctx = {
    ...sessionCtx(path.join(sandbox.dir, 'session.jsonl')),
    hasUI: true,
    ui: { setWidget: (id, lines) => drawn.push([id, lines]), notify: () => {} },
  };
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx, drawn };
}

/**
 * A conversation that GROWS at every round: the second compression has nothing to do on
 * the messages the first one already took, and would answer `nothing-to-compress`.
 */
const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

test('the widget draws the index shape and never enters the context', async () => {
  const { home, tools, hooks, ctx, drawn } = await boot();
  try {
    // Two leaves with a micro each, then one turn: with looseLeaves 1 and nodeCapacity 2
    // that is one node holding one leaf, plus one loose leaf.
    for (let i = 1; i <= 2; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BLOCK-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    await hook(hooks, ctx, conversation(1, 12));

    assert.ok(drawn.length > 0, 'the index shape was never drawn: showWidget is still read by nobody');
    const last = drawn.at(-1);
    assert.equal(last[0], 'cwl-index', `the widget has an unexpected id: ${last[0]}`);
    assert.match(
      String(last[1][0]),
      /pit \d+n\/\d+l │ topics \d+n\/\d+l │ buffer \d+n\/\d+l │ ordinary \d+n\/\d+l/,
      `the drawn line is not the index shape: ${last[1]}`,
    );
    assert.match(String(last[1][0]), /^CWL ▸ /, `the drawn line lost its prefix: ${last[1]}`);

    // The whole point: a window is not a message.
    const messages = await hook(hooks, ctx, conversation(1, 12));
    const leaked = messages.filter((m) => String(m.content ?? '').includes('CWL ▸'));
    assert.equal(leaked.length, 0, `the widget leaked into the context: ${JSON.stringify(leaked)}`);
  } finally {
    home.restore();
  }
});

test('showWidget: false clears the widget', async () => {
  const { home, tools, hooks, ctx, drawn } = await boot({ showWidget: false });
  try {
    await hook(hooks, ctx, conversation(1, 4));
    await tools.get('cwl_compress_range').execute(
      't', { summary: 'BLOCK-1 ' + 'x'.repeat(300) }, undefined, undefined, ctx,
    );
    await hook(hooks, ctx, conversation(1, 12));
    const cleared = drawn.filter(([id, lines]) => id === 'cwl-index' && lines === undefined);
    assert.ok(
      cleared.length > 0,
      `with showWidget false the widget was never cleared: ${JSON.stringify(drawn)}`,
    );
  } finally {
    home.restore();
  }
});
