#!/usr/bin/env bash
# Gate di pi-cwl: risolve le dipendenze, typecheck, poi la suite completa.
#
# Niente path assoluti: prima questo script cercava tsc in
# /home/riccardo/.hermes/..., quindi funzionava su una sola macchina e con un
# solo nome utente. Le dipendenze di Pi (typebox, pi-agent-core) si trovano a
# runtime e vengono collegate in node_modules/ (gitignorato: sono link derivati).
set -euo pipefail
cd "$(dirname "$0")"

# 1. Collega le dipendenze di Pi dentro node_modules/. Idempotente: se i link
#    esistono gia', non fa nulla. E' il passo che rende la suite portabile.
if ! node tests/_helpers.mjs --link-deps; then
  echo "Impossibile collegare le dipendenze di Pi." >&2
  echo "Imposta PI_PACKAGE_DIR=/path/al/pacchetto, oppure assicurati che 'pi' sia nel PATH." >&2
  exit 1
fi

# 2. Typecheck, se un tsc e' disponibile. Non e' obbligatorio: il gate vero sono
#    i test, che importano il modulo reale. Senza tsc si perde il controllo dei
#    tipi ma non la regressione.
TSC="${TSC:-}"
TYPECHECK=skipped
if [ -z "$TSC" ] && [ -x node_modules/.bin/tsc ]; then
  TSC="node_modules/.bin/tsc"
fi
if [ -z "$TSC" ] && command -v tsc >/dev/null 2>&1; then
  TSC="$(command -v tsc)"
fi
if [ -n "$TSC" ] && [ -x "$TSC" ]; then
  "$TSC" -p tsconfig.check.json "$@"
  TYPECHECK=run
else
  echo "[check.sh] tsc non trovato: salto il typecheck, eseguo i test." >&2
  echo "[check.sh] (installa typescript, o esporta TSC=/path/to/tsc, per il controllo dei tipi)" >&2
fi

# 3. Carica il modulo VERO. Il typecheck non puo' vedere un errore a runtime
#    durante la valutazione del modulo (temporal dead zone, binding non
#    inizializzato, import rotto), e un errore del genere uccide l'estensione in
#    silenzio: nessun tool registrato, nessun messaggio, typecheck verde.
#    Importa index.ts direttamente: Node >= 22.18 strippa i tipi da solo.
STATUS=0
node --test tests/*.test.mjs || STATUS=$?

# 4. Se il typecheck NON e' stato eseguito, dirlo in modo IMPOSSIBILE da non
#    vedere. Prima usciva solo su stderr, e un `grep` dei risultati lo
#    nascondeva: per tre giri ho letto "green" e ho creduto che i tipi fossero
#    controllati. Un errore vero e' passato cosi' (`st.tokenBudgetX`, un campo
#    che non esiste). Un gate che salta un controllo in silenzio e' peggio di un
#    gate che non lo fa.
if [ "$TYPECHECK" = "skipped" ]; then
  echo ""
  echo "================================================================"
  echo "ATTENZIONE: TYPECHECK SALTATO — nessun tsc disponibile."
  echo "Verde qui significa: i TEST passano. NON significa: i tipi sono a posto."
  echo "Per il controllo dei tipi: esporta TSC=/percorso/del/tsc, oppure"
  echo "installa typescript come devDependency."
  echo "================================================================"
fi
exit $STATUS
