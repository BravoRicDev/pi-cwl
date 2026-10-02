/**
 * Regression: the session state survives a restart, and two different
 * sessions do not overwrite each other.
 *
 * Why it exists. Three defects in the "session state" area:
 *
 * 1. `sessionKey` was `${cwd}::${sid}`. Without a session id it became `::` or
 *    the constant `default`, so TWO anonymous sessions in the same folder
 *    shared a single state.
 * 2. `session_start` did `states.set(key, newState())` without checking whether the
 *    state existed: the episode graph was zeroed at every start.
 * 3. The state lived only in RAM. The episode graph and the compressions
 *    tracked with `cwl_compress` are DECISIONS of the agent, not data
 *    recomputable from the transcript: lost at restart, the work was lost.
 *
 * The test drives the real hooks of the extension (session_start / delimiter /
 * cwl_compress / context / turn_end) and restarts the module from scratch, as a
 * restart of Pi would.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { createHash } from 'node:crypto';
import { makeSandbox, bootExtension, withHome, sessionCtx, anonymousCtx, status } from './_helpers.mjs';

/** Starts a new independent instance of the extension on the same sandbox. */
const reboot = (sandbox, tag) => bootExtension(sandbox, { name: tag });

/** Hash of the text of a message, the way hashText() in the extension computes it. */
const hashOf = (text) => createHash('sha256').update(text).digest('hex').slice(0, 12);

const msg = (text, role = 'assistant') => ({ role, content: text, timestamp: 1 });

test('the episode graph survives a restart', async () => {
  const sandbox = makeSandbox({ name: 'state-graph' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));

    // --- First run: open and close an episode.
    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.tools.get('delimiter').execute('tc2',
      { name: 'exploration-1', type: 'expl', description: 'learned X' }, undefined, undefined, session);
    await run1.hooks.get('turn_end')({}, session);
    assert.equal((await status(run1.tools, session)).details.total, 1, 'before the restart: 1 episode');

    // --- Second run: new module, same HOME, same session.
    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const after = await status(run2.tools, session);
    assert.equal(after.details.total, 1, "after the restart the episode must be found again, not lost");
    assert.equal(after.details.closed, 1, "the episode must stay closed, as it was before the restart");
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('tracked compressions survive a restart', async () => {
  const sandbox = makeSandbox({ name: 'state-span' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    // Both endpoints are ASSISTANT messages: `user` messages are
    // inviolable by design (Principle 3), so a user->user range
    // would keep the originals and would only ADD the summary.
    // With two assistant messages the span is really replaced and the proof is sharp.
    const all = [msg('first block of work'), msg('second block of work')];

    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    // The context hook registers the hashes of the messages present: without them
    // cwl_compress would reject the hash as unknown.
    await run1.hooks.get('context')({ messages: all }, session);

    const res = await run1.tools.get('cwl_compress').execute('tc3', {
      startHash: hashOf(all[0].content),
      endHash: hashOf(all[1].content),
      summary: 'summary of the pair',
    }, undefined, undefined, session);
    assert.equal(res.details?.ok, true, `the compression must be accepted (${res.content[0].text})`);
    assert.equal(res.details?.spans, 1);

    // --- Restart.
    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const messages = [msg('first block of work'), msg('second block of work')];
    const out = await run2.hooks.get('context')({ messages }, session);
    assert.ok(out?.messages, 'the resumed compression must be applied to the context');
    assert.ok(out.messages.length < messages.length,
      `the two messages must be replaced by the summary: ${out.messages.length} vs ${messages.length}`);
    const joined = JSON.stringify(out.messages);
    assert.match(joined, /summary of the pair/, 'the summary text must be in the context');
    assert.doesNotMatch(joined, /first block of work/, 'the original texts must have been replaced');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('a revoked compression does not rise from the disk', async () => {
  const sandbox = makeSandbox({ name: 'state-revoke' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    const pair = [msg('alpha'), msg('beta')];

    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.hooks.get('context')({ messages: pair }, session);

    const tool = run1.tools.get('cwl_compress');
    const args = { startHash: hashOf('alpha'), endHash: hashOf('beta'), summary: 'S' };
    await tool.execute('t1', args, undefined, undefined, session);
    const revoke = await tool.execute('t2', args, undefined, undefined, session);
    assert.equal(revoke.details?.revoked, true, 'the second call must revoke');

    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const out = await run2.hooks.get('context')({ messages: pair }, session);
    // No span: the context must not be touched (we are below threshold).
    assert.ok(!out?.messages || out.messages.length === pair.length,
      'a revoked compression must not be re-applied after the restart');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('two sessions without id do not share the state', async () => {
  const sandbox = makeSandbox({ name: 'state-anon' });
  const home = withHome(sandbox.dir);
  try {
    const { hooks, tools } = await bootExtension(sandbox);
    const a = anonymousCtx();
    const b = anonymousCtx();

    await hooks.get('session_start')({}, a);
    await hooks.get('session_start')({}, b);

    // An episode recorded in session A must not appear in B.
    await tools.get('delimiter').execute('tc1',
      { name: 'only-in-a', type: 'expl' }, undefined, undefined, a);
    assert.equal((await status(tools, a)).details.total, 1, 'A sees its own episode');
    assert.equal((await status(tools, b)).details.total, 0, 'B must NOT see the state of A');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('state files are pruned when they get old', async () => {
  const sandbox = makeSandbox({ name: 'state-prune' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    const run1 = await bootExtension(sandbox);
    await run1.hooks.get('session_start')({}, session);
    await run1.tools.get('delimiter').execute('tc1',
      { name: 'e', type: 'expl' }, undefined, undefined, session);
    await run1.hooks.get('turn_end')({}, session);

    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    const files = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(files.length, 1, 'one state file per session');

    // Ages the file beyond the retention threshold.
    const old = new Date(Date.now() - 20 * 24 * 60 * 60 * 1000);
    for (const f of files) fs.utimesSync(path.join(stateDir, f), old, old);

    const run2 = await reboot(sandbox, 'run2');
    await run2.hooks.get('session_start')({}, session);
    const left = fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'));
    assert.equal(left.length, 0, 'an old state file must be removed');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('a session that does not use CWL writes nothing to disk', async () => {
  const sandbox = makeSandbox({ name: 'state-idle' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    const { hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, session);
    await hooks.get('context')({ messages: [msg('hello')] }, session);
    await hooks.get('turn_end')({}, session);

    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    const files = fs.existsSync(stateDir)
      ? fs.readdirSync(stateDir).filter((f) => f.endsWith('.json'))
      : [];
    assert.equal(files.length, 0, 'without episodes or compressions there must be no state to save');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('a corrupt state does not prevent startup', async () => {
  const sandbox = makeSandbox({ name: 'state-corrupt' });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
    fs.mkdirSync(stateDir, { recursive: true });

    // Writes an unreadable state with the same name the extension would use.
    const key = session.sessionManager.getSessionFile();
    const name = `${createHash('sha256').update(key).digest('hex').slice(0, 32)}.json`;
    fs.writeFileSync(path.join(stateDir, name), '{ truncated halfway');

    const { hooks, tools } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, session);
    const st = await status(tools, session);
    assert.equal(st.details.total, 0, 'a corrupt file must degrade to an empty state, not make startup fail');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

/**
 * Regression: the debug log does not grow without limit.
 *
 * Why it exists. `debugLog` did `fs.mkdirSync` + `fs.appendFileSync` with no
 * cap at all. The state files have `pruneStateFiles()` (14 days), the log had
 * nothing. And the log turns on EXACTLY when chasing a bug in a
 * long session: it is the situation where it grows fastest, and nobody
 * notices until the disk fills up.
 *
 * The rotation keeps ONE generation (`MAX_LOG_BYTES = 2_000_000`, then rename to
 * `.1`): the file stays bounded to twice the cap and the previous session
 * stays readable instead of being thrown away.
 *
 * `LOG_PATH` is computed at IMPORT time of the module: if `withHome` arrived after
 * `bootExtension`, the test would write into the real home of the operator.
 */
test('the debug log rotates instead of growing without limit', async () => {
  const sandbox = makeSandbox({ name: 'debug-log', config: { debug: true } });
  const home = withHome(sandbox.dir);
  try {
    const session = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
    const logPath = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
    fs.mkdirSync(path.dirname(logPath), { recursive: true });

    // A log already above the cap, with recognizable content: so one can see
    // whether the old generation is preserved or lost.
    const oldLog = 'OLD LOG\n'.repeat(300000); // ~2.4 MB
    fs.writeFileSync(logPath, oldLog);
    assert.ok(fs.statSync(logPath).size > 2_000_000, 'the starting log must exceed the cap');

    const { hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, session);

    const rotatedLog = `${logPath}.1`;
    assert.ok(fs.existsSync(rotatedLog), 'the log above the cap must be rotated to .1');
    assert.equal(fs.readFileSync(rotatedLog, 'utf8'), oldLog,
      'the previous generation must stay intact, not be thrown away');
    const newLog = fs.readFileSync(logPath, 'utf8');
    assert.ok(newLog.includes('SESSION START'),
      `the new log must contain the startup line, found: ${newLog.slice(0, 200)}`);
    assert.ok(fs.statSync(logPath).size < 2_000_000, 'the new log must restart small');
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
