/**
 * Smoke load: l'estensione si carica e registra i suoi tool.
 *
 * Perche' esiste: il 29/09/2026 `const I18N = { ... t('params').description ... }`
 * (auto-riferimento dentro il literal I18N) faceva fallire il caricamento con
 * "Cannot access 'I18N' before initialization". Il typecheck era PULITO: l'errore
 * e' a runtime, in fase di valutazione del modulo. Nessuna suite lo copriva.
 * Qui si importa l'estensione vera: se il corpo del modulo esplode, il test esplode.
 *
 * Importa `index.ts` DIRETTAMENTE: Node >= 22.18 strippa i tipi da solo. Prima
 * questo file importava `typescript` con un path assoluto per fare il transpile
 * a mano, e la suite girava su una sola macchina.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { makeSandbox, bootExtension, withHome } from './_helpers.mjs';

test('il modulo si carica e registra i suoi tool', async () => {
  const sandbox = makeSandbox({ name: 'smoke' });
  const home = withHome(sandbox.dir);
  try {
    // Se il corpo del modulo lancia (TDZ, binding non inizializzato, import
    // rotto), questo boot lancia e il test fallisce con l'errore vero.
    const { tools } = await bootExtension(sandbox);

    for (const name of ['delimiter', 'cwl_status', 'cwl_compress', 'cwl_recall']) {
      assert.ok(tools.has(name), `tool "${name}" non registrato`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('le descrizioni dei parametri sono leggibili a runtime', async () => {
  const sandbox = makeSandbox({ name: 'smoke-desc' });
  const home = withHome(sandbox.dir);
  try {
    const { tools } = await bootExtension(sandbox);
    const delimiter = tools.get('delimiter');
    for (const param of ['action', 'name', 'type', 'dependencies', 'description']) {
      const p = delimiter.parameters?.properties?.[param];
      assert.ok(p, `parametro "${param}" assente da delimiter`);
      assert.equal(typeof p.description, 'string', `descrizione di "${param}" non e' una stringa`);
      assert.ok(p.description.length > 0, `descrizione di "${param}" vuota`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('ogni tool espone un nome e una descrizione non vuota', async () => {
  const sandbox = makeSandbox({ name: 'smoke-tools' });
  const home = withHome(sandbox.dir);
  try {
    const { tools } = await bootExtension(sandbox);
    for (const [name, tool] of tools) {
      assert.ok(typeof tool.description === 'string' && tool.description.length > 0,
        `tool "${name}" senza descrizione`);
      assert.ok(tool.parameters, `tool "${name}" senza schema dei parametri`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
