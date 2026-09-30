/**
 * IN AN AUTONOMOUS SESSION THE TURNS ARE NOT ONLY MINE.
 *
 * The operator's request, word for word: *"when I set the autonomous agents up
 * properly I do not write for entire days, the memory card and cronjob must count
 * as 'turns' exactly as if they were mine. Otherwise this will never work."*
 *
 * WHY HE WAS RIGHT. `protectedFromIndex` walked the list counting ONLY the `user`
 * messages: the window started from the last prompt the operator typed. In an
 * autonomous session that prompt is DAYS old, so everything that came after —
 * days of work, wake-ups, cards — ended up inside the protected window. And when
 * the list has fewer `user` turns than the window, the degenerate branch kicks in:
 * the floor goes to 80% of the list and NONE of the three compression routes can
 * touch anything anymore.
 * MEASURED live, in the operator's session: `CONTEXT 436837t ... 348650t in the
 * protected window` — 348,650 out of 436,837 is the 80%, that is, exactly that branch.
 *
 * THE FIX. A turn boundary is a `user` message OR an injection from ANOTHER
 * extension: a cronjob's wake-up (`background-task-notification`) and the card
 * (`anti-amnesia`) open a turn exactly like a prompt. OUR injections do not: a
 * summary stands where the compressed messages stood, and the index request is at the
 * BOTTOM of the list — counting them would move the window onto the wrong thing.
 *
 * THE TWO THINGS THE TEST DEMANDS:
 *  1. the protected window does NOT cover nearly everything (with the `user` messages alone it
 *     would cover 80%);
 *  2. the compression is POSSIBLE: `cwl_compress_range` must succeed. It is the proof that
 *     matters to the operator — "non resta niente da comprimere" is the symptom that
 *     led him to ask for this fix.
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
  protectedTurns: 4,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `turns-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

/** Sixteen autonomous turns after the operator's last prompt, which is days earlier. */
const autonomousList = () => {
  const out = [
    { role: 'user', content: 'operator, days ago ' + 'U'.repeat(300), timestamp: 1 },
    { role: 'assistant', content: 'the reply from back then ' + 'A'.repeat(300), timestamp: 2 },
  ];
  for (let i = 1; i <= 16; i++) {
    // A cronjob wake-up: for the extension it is a `custom` injection, but for the
    // conversation it is the start of a turn like a prompt.
    out.push({ role: 'custom', customType: 'background-task-notification', content: `wake-up ${i}`, timestamp: 100 + i * 3 });
    out.push({ role: 'assistant', content: `autonomous work ${i} ` + 'A'.repeat(300), timestamp: 101 + i * 3 });
    out.push({ role: 'toolResult', content: `output ${i} ` + 'T'.repeat(300), timestamp: 102 + i * 3 });
  }
  // And at the bottom one of OUR injections: the index request. It is not a turn: if it
  // counted, the floor would slip back by one turn every time it is asked for.
  out.push({ role: 'custom', customType: 'cwl-compressed', content: 'our summary', timestamp: 9000 });
  return out;
};

test('an autonomous turn (wake-up, card) counts as an operator turn', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const list = autonomousList();
    const before = logOf(sandbox).length;
    await hook(hooks, ctx, list);
    const log = logOf(sandbox).slice(before);

    const row = /CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window/.exec(log);
    assert.ok(
      row,
      `the turn does not declare the floor (the two figures are needed to measure): ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );
    const total = Number(row[1]);
    const protectedTokens = Number(row[2]);
    const share = protectedTokens / total;
    assert.ok(
      share < 0.5,
      `the protected window covers ${protectedTokens}t of ${total}t (${Math.round(share * 100)}%): with the \`user\` messages alone the ` +
        'floor slips onto the operator\'s last prompt — which in an autonomous session is days old — and almost all of the ' +
        'context stays protected. The wake-ups and the card must count as turns.',
    );

    // 2. And the proof that matters to the operator: compression must be possible.
    const range = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-A' }, undefined, undefined, ctx);
    assert.equal(
      range.details.ok,
      true,
      `"non resta niente da comprimere": it is the symptom that led to this fix — ${JSON.stringify(range.details)}`,
    );
  } finally {
    home.restore();
  }
});

/**
 * Ten exchanges, each of which OPENS with three consecutive injections.
 *
 * It is the real shape: at the start of an exchange a cronjob wake-up, a memory
 * card renewal and maybe another extension's notification arrive together. None of
 * them is an extra turn: they are the SAME start.
 */
const exchangesWithConsecutiveInjections = (exchanges) => {
  const out = [
    { role: 'user', content: 'operator, days ago ' + 'U'.repeat(300), timestamp: 1 },
    { role: 'assistant', content: 'the reply from back then ' + 'A'.repeat(300), timestamp: 2 },
  ];
  for (let i = 1; i <= exchanges; i++) {
    out.push({ role: 'custom', customType: 'background-task-notification', content: `wake-up ${i}`, timestamp: 100 + i * 10 });
    out.push({ role: 'custom', customType: 'anti-amnesia', content: `card ${i}`, timestamp: 101 + i * 10 });
    out.push({ role: 'custom', customType: 'background-task-notification', content: `another wake-up ${i}`, timestamp: 102 + i * 10 });
    out.push({ role: 'assistant', content: `autonomous work ${i} ` + 'A'.repeat(400), timestamp: 103 + i * 10 });
    out.push({ role: 'toolResult', content: `output ${i} ` + 'T'.repeat(400), timestamp: 104 + i * 10 });
  }
  return out;
};

test('three consecutive injections count as ONE single turn', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    const before = logOf(sandbox).length;
    await hook(hooks, ctx, exchangesWithConsecutiveInjections(10));
    const log = logOf(sandbox).slice(before);
    const row = /CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window/.exec(log);
    assert.ok(row, `the turn does not declare the floor: ${log.trim().split('\n').slice(-3).join(' | ')}`);
    const share = Number(row[2]) / Number(row[1]);
    assert.ok(
      share > 0.35,
      `the protected window covers only ${Math.round(share * 100)}% of the context: the ADJACENT boundaries are counting ` +
        'one by one, so the three injections that open the same exchange eat three "turns" and the window slips ' +
        'toward the present. Four turns must stay four exchanges, even when several tools fire in sequence.',
    );
  } finally {
    home.restore();
  }
});

/**
 * A case is needed where the difference is VISIBLE: with few turns, if our injections
 * counted as boundaries the floor would slip by four turns in one go.
 */
const shortList = () => {
  const out = [
    { role: 'user', content: 'operator, days ago ' + 'U'.repeat(300), timestamp: 1 },
    { role: 'assistant', content: 'the reply from back then ' + 'A'.repeat(300), timestamp: 2 },
  ];
  for (let i = 1; i <= 4; i++) {
    out.push({ role: 'custom', customType: 'background-task-notification', content: `wake-up ${i}`, timestamp: 100 + i * 3 });
    out.push({ role: 'assistant', content: `autonomous work ${i} ` + 'A'.repeat(400), timestamp: 101 + i * 3 });
    out.push({ role: 'toolResult', content: `output ${i} ` + 'T'.repeat(400), timestamp: 102 + i * 3 });
  }
  for (const ct of ['cwl-compressed', 'cwl-demand', 'cwl-compressed', 'cwl-demand']) {
    out.push({ role: 'custom', customType: ct, content: 'our injection', timestamp: 9000 });
  }
  return out;
};

test('OUR injections do not count as turns', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    const before = logOf(sandbox).length;
    await hook(hooks, ctx, shortList());
    const log = logOf(sandbox).slice(before);
    const row = /CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window/.exec(log);
    assert.ok(row, `the turn does not declare the floor: ${log.trim().split('\n').slice(-3).join(' | ')}`);
    const share = Number(row[2]) / Number(row[1]);
    assert.ok(
      share > 0.5,
      `the protected window covers only ${Math.round(share * 100)}% of the context: the four OUR injections at the bottom ` +
        'of the list are counting as turns, so the floor slips by four turns and the real autonomous turns stay ' +
        'uncovered. An index request or a summary do not open a turn: they stand where the compression stands, or at the bottom ' +
        'of the list because they are needed now.',
    );
  } finally {
    home.restore();
  }
});
