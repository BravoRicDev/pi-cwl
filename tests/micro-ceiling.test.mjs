/**
 * THE CEILING OF A LABEL, AND THE EXCESS TOLD OUTSIDE.
 *
 * MEASURED on 42 real labels: ~722t each (~1.900 characters) against the ~300t the design
 * took for granted. The top of the index is made of labels, so it cost 30k instead of
 * ~12k. The prompt said "~200 words" and constrained nothing: 42 labels came out at ~300
 * words each. A ceiling in CHARACTERS is something a model can count while writing.
 *
 * The excess is NOT rejected: rejecting it would block the work, and an overly long label
 * is still a label. It is DECLARED, in the tool result and in the log. What is not
 * acceptable is silence.
 *
 * Two cases, and both are needed: above the ceiling the declaration MUST be there; below the ceiling it
 * must NOT be, otherwise "declare the excess" would be indistinguishable from "always declare".
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;
const TETTO = 1400;

const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `tetto-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

/** Drives the hook (which computes the address) and then compresses with the requested label. */
async function comprimi(sandbox, hooks, ctx, tools, etichetta) {
  const base = conversazione(1, 7);
  await hooks.get('context')({ messages: base }, ctx);
  const prima = logDi(sandbox).length;
  const res = await tools
    .get('cwl_compress_range')
    .execute('t', { summary: 'BODY-1 ' + 'x'.repeat(300), micro: etichetta }, undefined, undefined, ctx);
  const testo = res.content.map((c) => c.text).join('\n');
  return { res, testo, dopo: logDi(sandbox).slice(prima) };
}

test("a label over the ceiling is RECORDED and the excess DECLARED in the result and in the log", async () => {
  const { sandbox, home, hooks, ctx, tools } = await boot();
  try {
    const { res, testo, dopo } = await comprimi(sandbox, hooks, ctx, tools, 'E'.repeat(TETTO + 100));
    assert.equal(res.details.ok, true, `the compression did not go through: ${JSON.stringify(res.details)}`);
    assert.match(
      testo,
      new RegExp(String(TETTO)),
      "the result does not name the ceiling: the agent does not know it blew the measure, and the next label will be just as long",
    );
    assert.match(
      dopo,
      /MICRO over ceiling: \S+ is \d+ characters \(~\d+t\) against a ceiling of \d+ \(~\d+t\)/,
      'the log did not measure the excess: no line, no number, no way to notice it',
    );
  } finally {
    home.restore();
  }
});

test("a label under the ceiling produces no declaration (measuring is not a ritual)", async () => {
  const { sandbox, home, hooks, ctx, tools } = await boot();
  try {
    const { res, testo, dopo } = await comprimi(sandbox, hooks, ctx, tools, 'E'.repeat(TETTO - 100));
    assert.equal(res.details.ok, true, `the compression did not go through: ${JSON.stringify(res.details)}`);
    assert.doesNotMatch(testo, new RegExp(String(TETTO)), 'the result names the ceiling even when it was not blown');
    assert.doesNotMatch(dopo, /MICRO over ceiling/, 'the log declares an excess that did not happen');
  } finally {
    home.restore();
  }
});
