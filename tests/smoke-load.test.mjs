/**
 * Smoke load: l'estensione si carica e registra i suoi tool.
 *
 * Perche' esiste: il 29/09/2026 `const I18N = { ... t('params').description ... }`
 * (auto-riferimento dentro il literal I18N) faceva fallire il caricamento con
 * "Cannot access 'I18N' before initialization". Il typecheck era PULITO: l'errore
 * e' a runtime, in fase di valutazione del modulo. Nessuna suite lo copriva.
 * Qui si importa l'estensione vera: se il corpo del modulo esplode, il test esplode.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from '/home/riccardo/.hermes/lsp/node_modules/typescript/lib/typescript.js';

const root = path.resolve(import.meta.dirname, '..');
const TYPEBOX = '/home/riccardo/.hermes/node/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/typebox/build/index.mjs';

test('the extension module loads and registers its tools', async () => {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'cwl-smoke-'));
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const source = fs.readFileSync(path.join(root, 'index.ts'), 'utf8');
    const js = ts
      .transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
      })
      .outputText.replace("from 'typebox'", `from '${pathToFileURL(TYPEBOX).href}'`);
    fs.writeFileSync(path.join(temp, 'index.mjs'), js);
    fs.copyFileSync(path.join(root, 'config.json'), path.join(temp, 'config.json'));

    // Se il corpo del modulo lancia (TDZ, riferimento a un binding non inizializzato,
    // import rotto), questo await lancia e il test fallisce con l'errore vero.
    const { default: extension } = await import(pathToFileURL(path.join(temp, 'index.mjs')).href);
    assert.equal(typeof extension, 'function', 'index.ts deve esportare una default function');

    const tools = new Map();
    const pi = {
      on() {},
      registerTool(tool) { tools.set(tool.name, tool); },
      registerCommand() {},
      sendMessage() {},
    };
    extension(pi);

    for (const name of ['delimiter', 'cwl_status', 'cwl_compress', 'cwl_recall']) {
      assert.ok(tools.has(name), `tool "${name}" non registrato`);
    }

    // Il catalogo i18n deve essere leggibile a runtime, non solo compilare.
    const delimiter = tools.get('delimiter');
    for (const param of ['action', 'name', 'type', 'dependencies', 'description']) {
      const p = delimiter.parameters?.properties?.[param];
      assert.ok(p, `parametro "${param}" assente da delimiter`);
      assert.equal(typeof p.description, 'string', `descrizione di "${param}" non e' una stringa`);
      assert.ok(p.description.length > 0, `descrizione di "${param}" vuota`);
    }
  } finally {
    process.env.HOME = home;
    fs.rmSync(temp, { recursive: true, force: true });
  }
});
