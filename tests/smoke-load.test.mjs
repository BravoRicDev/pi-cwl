/**
 * Smoke load: the extension loads and registers its tools.
 *
 * Why it exists: on 2026-09-29 `const I18N = { ... t('params').description ... }`
 * (self-reference inside the I18N literal) made the load fail with
 * "Cannot access 'I18N' before initialization". The typecheck was CLEAN: the error
 * is at runtime, during module evaluation. No suite covered it.
 * Here the real extension is imported: if the module body blows up, the test blows up.
 *
 * Imports `index.ts` DIRECTLY: Node >= 22.18 strips the types by itself. Before,
 * this file imported `typescript` with an absolute path to transpile
 * by hand, and the suite ran on a single machine.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { makeSandbox, bootExtension, withHome } from './_helpers.mjs';

test('the module loads and registers its tools', async () => {
  const sandbox = makeSandbox({ name: 'smoke' });
  const home = withHome(sandbox.dir);
  try {
    // If the module body throws (TDZ, uninitialized binding, broken
    // import), this boot throws and the test fails with the real error.
    const { tools } = await bootExtension(sandbox);

    for (const name of ['delimiter', 'cwl_status', 'cwl_compress', 'cwl_recall']) {
      assert.ok(tools.has(name), `tool "${name}" is not registered`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('the parameter descriptions are readable at runtime', async () => {
  const sandbox = makeSandbox({ name: 'smoke-desc' });
  const home = withHome(sandbox.dir);
  try {
    const { tools } = await bootExtension(sandbox);
    const delimiter = tools.get('delimiter');
    for (const param of ['action', 'name', 'type', 'dependencies', 'description']) {
      const p = delimiter.parameters?.properties?.[param];
      assert.ok(p, `parameter "${param}" missing from delimiter`);
      assert.equal(typeof p.description, 'string', `description of "${param}" is not a string`);
      assert.ok(p.description.length > 0, `description of "${param}" is empty`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});

test('every tool exposes a name and a non-empty description', async () => {
  const sandbox = makeSandbox({ name: 'smoke-tools' });
  const home = withHome(sandbox.dir);
  try {
    const { tools } = await bootExtension(sandbox);
    for (const [name, tool] of tools) {
      assert.ok(typeof tool.description === 'string' && tool.description.length > 0,
        `tool "${name}" without a description`);
      assert.ok(tool.parameters, `tool "${name}" without a parameter schema`);
    }
  } finally {
    home.restore();
    sandbox.cleanup();
  }
});
