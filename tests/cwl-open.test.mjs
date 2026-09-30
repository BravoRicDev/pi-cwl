/**
 * `cwl_open`: a leaf reopens WHOLE, and if it is not there it says so.
 *
 * THE PROBLEM IT SOLVES. A summary lives ONLY in the state: the transcript is
 * append-only and holds the ORIGINAL MESSAGES, not the summaries. As long as the only way
 * to reread what a compression had set aside was to buy the messages back
 * from the transcript, the agent's work (the summary) was writable and
 * not rereadable. `cwl_open` closes that circle.
 *
 * TWO RULES, and both are the operator's requests:
 *  1. the body comes back WHOLE, without truncation. It is not a convenience detail:
 *     his models have 1M-token windows and an agent that wants to see
 *     something must SEE it. The tool's only duty is to declare the size
 *     BEFORE handing it over, not to cut it;
 *  2. an id that does not exist is DECLARED. Silence is how bugs
 *     hide, and here it would be the worst: the agent would believe it had the
 *     content in hand and it does not.
 *
 * The second case is what makes the first honest: without it, a `cwl_open` that
 * answers "here is the body" to a nonexistent id would pass the test.
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
  const sandbox = makeSandbox({ name: `open-${seq++}`, config: config() });
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

const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `round ${i} paragraph ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `response ${i} paragraph ` + 'A'.repeat(200) });
  }
  return out;
};

const testo = (res) => res.content.map((c) => c.text).join('\n');

test('cwl_open returns the WHOLE body and declares an id that does not exist', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await hook(hooks, ctx, conversazione());

    // A body far longer than any preview: if the tool truncates, it shows.
    const corpo = 'ENTIRE-SUMMARY-BODY ' + 'x'.repeat(8000) + ' BODY-ENDS-HERE';
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: corpo }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the span was not created: ${JSON.stringify(comp.details)}`);

    const sp = statoDi(sandbox).spans[0];
    assert.ok(sp && typeof sp.id === 'string' && sp.id.length > 0, `the span has no stable id: ${JSON.stringify(sp)}`);

    const res = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    const out = testo(res);
    assert.equal(res.details.ok, true, `cwl_open refused an id that exists: ${JSON.stringify(res.details)}`);

    // 1. The body comes back WHOLE: not "an extract", not "the first N characters".
    assert.ok(
      out.includes(corpo),
      `the body did not come back whole: the tool answered with ${out.length} characters, the body has ${corpo.length}. ` +
        'A truncated summary is a lost summary: the original text sits in the transcript, the summary does not.',
    );

    // 2. The size is declared BEFORE handing it over (the only duty).
    assert.match(out, /~\d+ token/, `the tool does not declare the body size: ${out.slice(0, 120)}`);

    // 3. A nonexistent id is DECLARED, not kept silent.
    const nessuno = await tools.get('cwl_open').execute('t', { id: 'sp-00000000' }, undefined, undefined, ctx);
    assert.equal(
      nessuno.details.ok,
      false,
      'a nonexistent id was accepted: the agent would believe it had in hand a content that does not exist',
    );
    assert.match(
      testo(nessuno),
      /sp-00000000/,
      `the declaration does not name the requested id: ${testo(nessuno).slice(0, 160)}`,
    );

    // 4. The leaf must be REACHABLE: the id the agent sees in the context
    //    (the notice injected in place of the compressed messages) must be the one
    //    `cwl_open` accepts. An id nobody sees is a tool nobody can
    //    use, and no test would notice until one really tries.
    const conAvviso = await hook(hooks, ctx, conversazione());
    const iniettato = conAvviso.find((m) => m && m.customType === 'cwl-compressed');
    assert.ok(iniettato, 'no compression message in the context: the leaf is not even named');
    const visto = /(sp-[0-9a-f]{8})/.exec(String(iniettato.content ?? ''));
    assert.ok(
      visto,
      `the notice in the context does not say which id to open: ${String(iniettato.content).slice(0, 140)}`,
    );
    const riaperto = await tools.get('cwl_open').execute('t', { id: visto[1] }, undefined, undefined, ctx);
    assert.equal(
      riaperto.details.ok,
      true,
      `the id seen in the context (${visto[1]}) is not openable: ${JSON.stringify(riaperto.details)}`,
    );
  } finally {
    home.restore();
  }
});
