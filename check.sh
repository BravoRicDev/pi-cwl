#!/usr/bin/env bash
# pi-cwl gate: resolves the dependencies, typechecks, then runs the full suite.
#
# No absolute paths: this script used to look for tsc in
# /home/riccardo/.hermes/..., so it worked on one machine only and with one
# user name. Pi's dependencies (typebox, pi-agent-core) are found at runtime
# and linked into node_modules/ (gitignored: they are derived links).
set -euo pipefail
cd "$(dirname "$0")"

# 1. Link Pi's dependencies into node_modules/. Idempotent: if the links are
#    already there, it does nothing. This is the step that makes the suite
#    portable.
if ! node tests/_helpers.mjs --link-deps; then
  echo "Cannot link Pi's dependencies." >&2
  echo "Set PI_PACKAGE_DIR=/path/to/package, or make sure 'pi' is in PATH." >&2
  exit 1
fi

# 2. Typecheck, when a tsc is available. It is not mandatory: the real gate is
#    the suite, which imports the real module. Without tsc the types go
#    unchecked, but regressions are still caught.
TSC="${TSC:-}"
TYPECHECK=skipped
if [ -z "$TSC" ] && [ -x node_modules/.bin/tsc ]; then
  TSC="node_modules/.bin/tsc"
fi
if [ -z "$TSC" ] && command -v tsc >/dev/null 2>&1; then
  TSC="$(command -v tsc)"
fi
DECLARED=no
if grep -q '"typescript"' package.json 2>/dev/null; then DECLARED=yes; fi
if [ -n "$TSC" ] && [ -x "$TSC" ]; then
  "$TSC" -p tsconfig.check.json "$@"
  TYPECHECK=run
  # A POSITIVE statement. That the types were checked must not be deduced from
  # the absence of a warning: the absence of a warning is exactly what was read
  # as "types are fine" for three rounds.
  echo "[check.sh] typecheck ran: $TSC -p tsconfig.check.json"
else
  echo "[check.sh] tsc not found: skipping the typecheck, running the tests." >&2
  # typescript is declared in package.json since the first commit, but
  # node_modules/ was populated for years ONLY by the --link-deps links: the
  # dependency was declared and not installed. This case must stay distinct
  # from "this machine has no compiler": here the remedy is a command, not a
  # decision.
  if [ "$DECLARED" = "yes" ]; then
    echo "[check.sh] WARNING: typescript is DECLARED in package.json and NOT installed." >&2
    echo "[check.sh] The remedy is 'npm install' (then 'node tests/_helpers.mjs --link-deps')." >&2
  fi
  echo "[check.sh] (or export TSC=/path/to/tsc for type checking)" >&2
fi

# 3. Load the REAL module. The typecheck cannot see a runtime error while the
#    module is being evaluated (temporal dead zone, uninitialised binding,
#    broken import), and such an error kills the extension silently: no tool
#    registered, no message, green typecheck.
#    Import index.ts directly: Node >= 22.18 strips the types by itself.
STATUS=0
node --test tests/*.test.mjs || STATUS=$?

# 4. If the typecheck did NOT run, say it in a way IMPOSSIBLE to miss. It used
#    to go out on stderr only, and a `grep` of the results hid it: for three
#    rounds I read "green" and believed the types had been checked. A real
#    error slipped through this way (`st.tokenBudgetX`, a field that does not
#    exist). A gate that skips a check silently is worse than a gate that does
#    not have it.
if [ "$TYPECHECK" = "skipped" ]; then
  echo ""
  echo "================================================================"
  echo "WARNING: TYPECHECK SKIPPED — no tsc available."
  echo "Green here means: the TESTS pass. It does NOT mean: the types are fine."
  if [ "$DECLARED" = "yes" ]; then
    echo "typescript is DECLARED in package.json and not installed: run 'npm install'."
  fi
  echo "For type checking: 'npm install', or export TSC=/path/to/tsc."
  echo "================================================================"
fi
exit $STATUS
