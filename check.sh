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
DECLARED=no
if grep -q '"typescript"' package.json 2>/dev/null; then DECLARED=yes; fi
if [ -n "$TSC" ] && [ -x "$TSC" ]; then
  "$TSC" -p tsconfig.check.json "$@"
  TYPECHECK=run
  # Dichiarazione POSITIVA. Che i tipi siano stati controllati non deve dedursi
  # dall'assenza di un avviso: l'assenza di un avviso e' esattamente cio' che per
  # tre giri e' stata letta come "tipi a posto".
  echo "[check.sh] typecheck eseguito: $TSC -p tsconfig.check.json"
else
  echo "[check.sh] tsc non trovato: salto il typecheck, eseguo i test." >&2
  # typescript e' dichiarato in package.json dal primo commit, ma node_modules/
  # per anni e' stato popolato SOLO dai link di --link-deps: la dipendenza era
  # dichiarata e non installata. Questo caso va distinto da "su questa macchina
  # il compilatore non c'e'": qui il rimedio e' un comando, non una decisione.
  if [ "$DECLARED" = "yes" ]; then
    echo "[check.sh] ATTENZIONE: typescript e' DICHIARATO in package.json e NON e' installato." >&2
    echo "[check.sh] Il rimedio e' 'npm install' (poi 'node tests/_helpers.mjs --link-deps')." >&2
  fi
  echo "[check.sh] (oppure esporta TSC=/path/to/tsc per il controllo dei tipi)" >&2
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
  if [ "$DECLARED" = "yes" ]; then
    echo "typescript e' DICHIARATO in package.json e non e' installato: lancia 'npm install'."
  fi
  echo "Per il controllo dei tipi: 'npm install', oppure esporta TSC=/percorso/del/tsc."
  echo "================================================================"
fi
exit $STATUS
