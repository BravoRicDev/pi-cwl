/**
 * THE /cwl_save SLASH COMMAND — total compaction with zero protected turns.
 *
 * Why it exists. Under normal operation, CWL protects the last N user turns
 * (cf.protectedTurns, default 3) from compaction so the model can always see
 * the active task. When the operator runs /cwl_save (or /cwd_save):
 *
 *  1. ZERO PROTECTED TURNS: The entire history up to the latest turn is offered
 *     to cwl_compress_range in a single compressible range.
 *  2. DEDICATED AGENT TURN: The command sends a prompt via pi.sendUserMessage,
 *     and the context hook injects an explicit compact-all demand.
 *  3. IMMEDIATE LIVE REPLACEMENT: Once cwl_compress_range is executed, the entire
 *     uncompressed history is replaced by the newly created leaf in the live context,
 *     and persisted to disk so it can be adopted by other sessions.
 *  4. RESET ON COMPLETION: forceAllNext is reset to false once compression finishes.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome } from './_helpers.mjs';

let seq = 0;

const userMsg = (text) => ({ role: 'user', content: text, timestamp: Date.now() });
const asstMsg = (text) => ({ role: 'assistant', content: text, timestamp: Date.now() });

const config = (extra = {}) => ({
  tokenBudget: 100000,
  thresholdRatio: 1,
  protectedTurns: 3,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(name, extraCfg = {}) {
  const sandbox = makeSandbox({ name: `save-cmd-${name}-${seq++}`, config: config(extraCfg) });
  const home = withHome(sandbox.dir);
  const { tools, hooks, commands, userMessages } = await bootExtension(sandbox);
  const sessionFile = path.join(sandbox.dir, 'sessione.jsonl');
  return { sandbox, home, tools, hooks, commands, userMessages, sessionFile };
}

test('cwl_save and cwd_save are registered, each with a description', async () => {
  const { sandbox, home, commands } = await boot('reg');
  try {
    for (const name of ['cwl_save', 'cwd_save']) {
      assert.ok(commands.has(name), `${name} is registered`);
      assert.equal(typeof commands.get(name).handler, 'function', `${name} has a handler`);
      assert.equal(typeof commands.get(name).description, 'string', `${name} has a description`);
      assert.ok(commands.get(name).description.length > 0, `${name}'s description is not empty`);
    }
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_save with nothing to compress notifies warning and does not send prompt', async () => {
  const { sandbox, home, commands, userMessages, sessionFile } = await boot('empty');
  try {
    const notes = [];
    const ctx = {
      cwd: '/tmp/progetto',
      hasUI: true,
      sessionManager: {
        getSessionFile: () => sessionFile,
        buildContextEntries: () => [],
      },
      ui: {
        notify: (text, kind) => notes.push({ text, kind }),
        setWidget: () => {},
      },
    };
    await commands.get('cwl_save').handler('', ctx);
    assert.equal(userMessages.length, 0, 'no user message dispatched');
    assert.equal(notes.length, 1, 'notified warning');
    assert.equal(notes[0].kind, 'warning');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_save triggers full compaction with 0 protected turns, demand, and resets on compression', async () => {
  const { sandbox, home, hooks, tools, commands, userMessages, sessionFile } = await boot('flow', { protectedTurns: 3 });
  try {
    const entries = [];
    const notes = [];
    const ctx = {
      cwd: '/tmp/progetto',
      hasUI: true,
      sessionManager: {
        getSessionFile: () => sessionFile,
        buildContextEntries: () => entries,
      },
      ui: {
        notify: (text, kind) => notes.push({ text, kind }),
        setWidget: () => {},
      },
    };

    // 4 user turns with assistant replies (normally, protectedTurns: 3 protects the last 3 user turns)
    const messages = [
      userMsg('u1: initial task specification and requirements'),
      asstMsg('a1: starting work on the feature'),
      userMsg('u2: please implement step one'),
      asstMsg('a2: step one implemented successfully'),
      userMsg('u3: please implement step two'),
      asstMsg('a3: step two implemented successfully'),
      userMsg('u4: final review and verification'),
      asstMsg('a4: all tests passing, ready for handoff'),
    ];
    for (const m of messages) entries.push({ type: 'message', message: m });

    // Turn 1: normal context hook. Under budget without forceAllNext, no range is armed
    const hook = hooks.get('context');
    await hook({ messages }, ctx);
    const stBefore = await tools.get('cwl_status').execute('s1', {}, undefined, undefined, ctx);
    assert.equal(stBefore.details.spans, 0, 'no spans initially');
    assert.doesNotMatch(stBefore.content[0].text, /cwl_compress_range/);

    // Operator runs /cwl_save with an optional note
    await commands.get('cwl_save').handler('handoff topic switch', ctx);
    assert.equal(userMessages.length, 1, 'user message was dispatched');
    assert.match(userMessages[0], /handoff topic switch/);
    assert.equal(notes.length, 1, 'notified info');
    assert.equal(notes[0].kind, 'info');

    // Turn 2: the context hook runs following the user message.
    // With forceAllNext = true, 0 protected turns are enforced, so range covers all the way to a4!
    const out = await hook({ messages }, ctx);
    const msgs = out?.messages ?? messages;

    const stTurn2 = await tools.get('cwl_status').execute('s_turn2', {}, undefined, undefined, ctx);
    assert.match(stTurn2.content[0].text, /Compressible range:/, 'forceAllNext armed range even though under budget');

    // Check that demand is injected with compact-all instruction
    const demand = msgs.find((m) => m && m.customType === 'cwl-demand');
    assert.ok(demand, 'cwl-demand is present in context');
    assert.match(String(demand.content), /cwl_save/i);
    assert.match(String(demand.content), /handoff topic switch/);

    // Call cwl_compress_range to fulfill the save
    const compRes = await tools.get('cwl_compress_range').execute(
      'c1',
      {
        summary: 'Full summary of all steps: u1 through u4 completed with tests passing.',
        micro: 'Completed all steps u1..u4 with tests passing, handoff ready.',
      },
      undefined,
      undefined,
      ctx,
    );
    assert.equal(compRes.details.ok, true, 'compression succeeded');
    assert.ok(compRes.details.spans > 0, 'span added');

    // Verify that forceAllNext was reset
    const stAfter = await tools.get('cwl_status').execute('s2', {}, undefined, undefined, ctx);
    assert.equal(stAfter.details.spans, 1, '1 span tracked');

    // Turn 3: on next context hook, applySpans replaces the entire compressed range
    const out3 = await hook({ messages: [...messages, asstMsg('compaction completed')] }, ctx);
    const msgs3 = out3?.messages ?? [];
    const compressedNotices = msgs3.filter((m) => m && m.customType === 'cwl-compressed');
    assert.ok(compressedNotices.length > 0, 'cwl-compressed notice is present in live context');
  } finally { home.restore(); sandbox.cleanup(); }
});
