# pi-cwl — Context Window Lifecycle

Structured context eviction for long-horizon Pi agents, implementing the technique described in
["Beyond Compaction: Structured Context Eviction for Long-Horizon Agents"](https://arxiv.org/abs/2606.11213).

**English** · [Italiano](#italiano) · Companion extension: [pi-arc](https://github.com/BravoRicDev/pi-arc)

---

## English

### The problem

Fixed-threshold context compaction (summarisation) breaks down in three specific ways for autonomous agents:

1. **Loss of critical detail.** Code specifics, build errors and requirements get flattened into prose.
2. **Summary-induced hallucination.** The model summarises its own history poorly — exactly when the
   context is already under pressure.
3. **Mid-turn interruption.** Compaction blocks while the agent is mid-chain of tool calls.

### What this extension does

CWL treats the session as a **graph of typed episodes** that the agent labels itself through the
`delimiter` tool. When the token budget is exceeded, a **deterministic, LLM-free policy** applies
graduated eviction, ordered by recoverability and causal dependency.

A compressed range does not disappear: it becomes a **leaf** carrying a short label, and labels are
gathered into **nodes** (up to 30 leaves each) and eventually into an **old node** that keeps the
oldest history. What the context shows is the index — labels, not bodies — and bodies stay
recoverable on demand: `cwl_open` returns one in full (never truncated), `cwl_micro` swaps a body
for its label, `cwl_old` folds the oldest nodes together. Nothing is lost, because every leaf can
still be reopened down to the original messages in the session transcript.

### Design principles

- **Zero LLM overhead.** Compression never calls the model. It is purely algorithmic and structural —
  no cost, no added hallucination, no blocking.
- **User and system messages are protected from CWL eviction.** This describes CWL's eviction pass; it does not override Pi's native compaction behavior.
- **Typed episode separation:**
  - `expl` (exploration) — searches, listings, orientation reads. On close, the agent supplies a
    concise description: *this is the only content that survives eviction*.
  - `act` (action) — writes, edits, command execution. Effects are persistent in the environment,
    so these are the first candidates for graduated eviction.
- **Causal dependencies.** An exploration episode cannot be evicted while an action that depends on
  it is still live in context.

### Eviction levels

Applied in ascending order of aggressiveness until the budget is satisfied:

| Level | Effect |
|---|---|
| `stripReasoning` | Drops thinking blocks, keeps tool calls and results |
| `stripBulkOutput` | Drops large enumerable tool output (grep, glob, listings) |
| `stripIntermediate` | Drops intermediate read/bash results, keeps conclusions |
| `removeEpisode` | Removes the episode entirely; `expl` episodes leave their description behind |

Each level is independently toggleable in `levels`.

### Tools

**`delimiter`** — records and closes ONE episode: the segment of work since the previous
delimiter (or since the start of the session). The episode opens **implicitly**: there is no
separate opening call.

| Parameter | Type | Notes |
|---|---|---|
| `name` | `string` | Required. Unique: it is how the eviction and `cwl_recall_episode` refer to the segment |
| `type` | `"expl"` \| `"act"` | Required. `expl` = exploration (reads, searches: not needed after the inference), `act` = a change with persistent effects |
| `dependencies` | `string[]` | Names of closed `expl` episodes an `act` builds on |
| `description` | `string` | What you learned. For an `expl` it is the only content that survives eviction |

**`cwl_status`** — current token budget, measured context size, episode counts, evictions performed
and tokens saved.

**`cwl_compress`** / **`cwl_compress_range`** — compress an explicit range of the conversation into
a summary. `cwl_compress_range` picks the oldest eligible interval for you, and accepts an optional
`micro` (the label that will represent the leaf in the index).

**`cwl_open`** — reopen a leaf: returns the full body, never truncated, and declares its size first.

**`cwl_micro`** — replace a leaf body with a short label (ceiling: 1,200 characters, ~300 tokens);
an empty `text` gives the body back.

**`cwl_old`** — fold the oldest young nodes into the old node: the summary of summaries.

**`cwl_group`** — groups leaves into a **topic node**, which is born collapsed: its `name` and
`description` stand for its contents in the index, so the description must already cover the
future use of the topic. The tool accepts `leaves`, `nodes`, `node`, `name`, `description`, and
`pit`. Only leaves in the buffer — the last node, attached to leaves still open — can be moved;
the buffer itself is never absorbed.

With `nodes`, a topic can contain later topic nodes, preserving chronological order: a parent can
absorb only nodes that follow it, and at the live frontier only consecutive topics can be nested.
The parent's description cannot be rewritten there. With `pit: true`, leaves can instead be
catalogued inside the old node; creating a pit topic requires at least three leaves. Inside the old
node, a topic description can be rewritten without changing the index or the pit synthesis. Adding
leaves to an existing topic (`node`, without `description`) does not rewrite its description.

A nested topic's injected block includes a code-generated shape line, for example
`(holds 3 node(s), 41 leaf/leaves in all — open them with cwl_open("<id>"))`. The line reports
contained nodes and total leaves; use `cwl_open` on a child id to inspect it.

The index is nested: the **old node** at the head, then **legacy and topic nodes**, then the
**buffer**, then loose/open leaves. When the configured node threshold is reached (three nodes by default), the old node can absorb
the two oldest eligible nodes, of either kind, if the size and savings guards also pass. Topic descriptions absorbed into it are
preserved word for word; the pit synthesis is a frozen snapshot, while later additions remain
visible through their topic. Superseded pit syntheses remain readable with `cwl_open` using ids such
as `<pit id>.s1`.

**`cwl_recall`** / **`cwl_recall_episode`** — find compressed text by keyword, or retrieve an evicted
episode by name, straight from the transcript.

### Installation

Pi loads the extension from `index.ts` at the repository root. Link or copy the extension into your
Pi extensions directory:

```
~/.pi/agent/extensions/pi-cwl  ->  <this repository>
```

### Configuration

Optional, at `~/.pi/cwl/config.json`:

```json
{
  "tokenBudget": 80000,
  "thresholdRatio": 0.85,
  "levels": {
    "stripReasoning": true,
    "stripBulkOutput": true,
    "stripIntermediate": true,
    "removeEpisode": true
  },
  "protectedTurns": 4,
  "showWidget": true,
  "debug": false
}
```

- `tokenBudget` — active token budget. `80000` is roughly 30% of a 256k context window. Raise it for
  larger windows.
- `thresholdRatio` — eviction triggers at `tokenBudget × thresholdRatio`.
- `protectedTurns` — how many of the most recent turn boundaries are inviolable (default `4`). Each
  boundary also carries everything that follows it, wake-ups and memory cards included, so this is
  the single knob that decides how much context can actually be compressed.
- `looseLeaves` — how many leaves stay loose, fully visible, before one is absorbed into a node
  (default `5`).
- `nodeCapacity` — how many leaves a node holds before it is full (default `30`).
- `mergeNodesAt` — how many young nodes trigger a merge into the old node (default `3`).
- `mergeMinRatio` — how many times the synthesis a merge frees must outweigh the one it writes (default `1.8`). It is the guard a merge AND a topic both pass: the topic floor is `ratio × 3,600` characters, never less than `mergeMinChars`, so with the defaults a topic must free at least `6,480` characters. Raising it makes topics rarer and merges harder; lowering it does the opposite for both.
- `mergeMinChars` — the absolute floor, in characters: below it a merge is refused whatever the ratio says (default `6000`).
- `topicInviteAt` — how many leaves the buffer may hold before the agent is invited to open a topic with `cwl_group` (default `18`). The buffer is the LAST node and the only one whose leaves can still be moved: a leaf that has entered a node can never be moved again, so this invitation is the last moment a topic can be born. It fires only when the group would pass the size guard, and it carries every number needed to act — the node id, how many leaves it holds, the characters of micro they carry, and the characters the guard requires.
- `gate` — enables the contextual request to compress older material when the budget remains exceeded and deterministic eviction has nothing left to remove (default `true`).
- `showWidget` — draws the index shape (pit / young / topics / loose / waiting / head) in one TUI line below the editor. A widget is UI: it never enters the context and costs no tokens (default `true`).

Missing keys fall back to defaults, so a partial file is valid.

### Development checks

Run `bash check.sh` to link Pi dependencies and execute the full test suite. Typechecking runs only when `tsc` is available; look for `[check.sh] typecheck ran:` in the output. If it is absent, the typecheck was skipped.

### Attribution

This is an **original, independent implementation** written in TypeScript for the Pi agent ecosystem.
It contains no code copied, forked or adapted from any third-party repository.

- **Paper:** "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents"
  — Andrew Semenov, Svyatoslav Dorofeev (2026), [arXiv:2606.11213](https://arxiv.org/abs/2606.11213)
- The CWL concept originates with the paper authors; this implementation is ours.
- See `CITATION.cff` for formal citation and `LICENSE` for terms.

---

## Italiano

### Il problema

La compattazione a soglia fissa del contesto (summarisation) fallisce in tre modi precisi per gli
agenti autonomi:

1. **Perdita di dettaglio critico.** Specifiche di codice, errori di compilazione e requisiti vengono
   appiattiti in prosa.
2. **Allucinazione indotta dalla sintesi.** Il modello riassume male il proprio passato — proprio
   quando il contesto è già sotto pressione.
3. **Interruzione a metà turno.** La compattazione blocca mentre l'agente è nel mezzo di una catena
   di chiamate.

### Cosa fa questa estensione

CWL tratta la sessione come un **grafo di episodi tipizzati** che l'agente etichetta da sé tramite
il tool `delimiter`. Quando il budget di token viene superato, una **politica deterministica e
LLM-free** applica un'eviction graduata, ordinata per recuperabilità e dipendenza causale.

Un intervallo compresso non sparisce: diventa una **foglia** che porta con sé un'etichetta breve.
Le etichette si raccolgono in **nodi** (fino a 30 foglie l'uno) e prima o poi in un **nodo vecchio**
che conserva la storia più antica. Quello che il contesto mostra è l'indice — le etichette, non i
corpi — e i corpi restano recuperabili a richiesta: `cwl_open` ne restituisce uno intero (mai
troncato), `cwl_micro` scambia un corpo con la sua etichetta, `cwl_old` accorpa i nodi più vecchi.
Niente va perso, perché ogni foglia si può riaprire fino ai messaggi originali nel transcript della
sessione.

### Principi di design

- **Zero LLM overhead.** La compressione non chiama mai il modello. È puramente algoritmica e
  strutturale — nessun costo, nessuna allucinazione aggiuntiva, nessun blocco.
- **CWL protegge i messaggi utente e di sistema dalla propria eviction.** Questo descrive la politica CWL e non modifica il comportamento della compattazione nativa di Pi.
- **Separazione episodica tipizzata:**
  - `expl` (esplorazione) — ricerche, listing, letture di orientamento. Alla chiusura, l'agente
    fornisce una descrizione concisa: *è questo l'unico contenuto che sopravvive all'eviction*.
  - `act` (azione) — scritture, edit, esecuzione di comandi. Gli effetti sono persistenti
    nell'ambiente, quindi sono i primi candidati all'eviction graduata.
- **Dipendenze causali.** Un episodio esplorativo non può essere evictato finché un'azione che da
  lui dipende è ancora attiva nel contesto.

### Livelli di eviction

Applicati in ordine di aggressività crescente fino a soddisfare il budget:

| Livello | Effetto |
|---|---|
| `stripReasoning` | Rimuove i blocchi di thinking, mantiene tool call e risultati |
| `stripBulkOutput` | Rimuove output di tool enumerabili di grandi dimensioni (grep, glob, listing) |
| `stripIntermediate` | Rimuove risultati intermedi di letture/bash, mantiene le conclusioni |
| `removeEpisode` | Rimuove l'episodio intero; gli episodi `expl` lasciano la loro descrizione |

Ogni livello è attivabile/disattivabile indipendentemente in `levels`.

### Tool

**`delimiter`** — registra e chiude UN episodio: il segmento di lavoro dal delimiter
precedente (o dall'inizio della sessione). L'episodio si apre **implicitamente**: non esiste
una chiamata di apertura separata.

| Parametro | Tipo | Note |
|---|---|---|
| `name` | `string` | Obbligatorio. Univoco: e' come l'eviction e `cwl_recall_episode` si riferiscono al segmento |
| `type` | `"expl"` \| `"act"` | Obbligatorio. `expl` = esplorazione (letture, ricerche: non serve dopo l'inferenza), `act` = una modifica con effetti persistenti |
| `dependencies` | `string[]` | Nomi degli episodi `expl` chiusi su cui un `act` si basa |
| `description` | `string` | Cosa hai imparato. Per un `expl` e' l'unico contenuto che sopravvive all'eviction |

**`cwl_status`** — budget corrente, dimensione misurata del contesto, conteggio episodi, eviction
eseguite e token risparmiati.

**`cwl_compress`** / **`cwl_compress_range`** — comprimono un intervallo esplicito della
conversazione in un riassunto. `cwl_compress_range` sceglie per te l'intervallo più vecchio
disponibile e accetta un `micro` opzionale: l'etichetta che rappresenterà la foglia nell'indice.

**`cwl_open`** — riapre una foglia: restituisce il corpo intero, mai troncato, e ne dichiara prima la
dimensione.

**`cwl_micro`** — sostituisce il corpo di una foglia con un'etichetta breve (obiettivo: ~960
caratteri, ~240 token; il tetto misurato è 1.400 caratteri, ~350 token); un `text` vuoto
restituisce il corpo.

**`cwl_old`** — accorpa i nodi giovani più vecchi nel nodo vecchio: il riassunto dei riassunti.

**`cwl_group`** — raggruppa le foglie in un **nodo topic**, che nasce collassato: `name` e
`description` rappresentano il contenuto nell'indice, quindi la descrizione deve coprire già l'uso
futuro del topic. Il tool accetta `leaves`, `nodes`, `node`, `name`, `description` e `pit`. Si possono
spostare solo foglie nel buffer — l'ultimo nodo, attaccato alle foglie ancora aperte; il buffer non
viene mai assorbito.

Con `nodes`, un topic può contenere topic successivi, mantenendo l'ordine cronologico: un genitore
può assorbire solo nodi che vengono dopo di lui e, alla frontiera attiva, solo topic consecutivi.
Alla frontiera la descrizione del genitore non si può riscrivere. Con `pit: true`, le foglie possono
essere catalogate dentro il nodo vecchio; per creare un topic nel pozzo servono almeno tre foglie.
Dentro il nodo vecchio la descrizione di un topic si può riscrivere senza cambiare l'indice o la
sintesi del pozzo. Aggiungere foglie a un topic esistente (`node`, senza `description`) non ne
riscrive la descrizione.

Il blocco iniettato di un topic annidato include una riga di forma generata dal codice, ad esempio
`(holds 3 node(s), 41 leaf/leaves in all — open them with cwl_open("<id>"))`. La riga indica i
nodi contenuti e il totale delle foglie; usa `cwl_open` con l'id del figlio per ispezionarlo.

L'indice è annidato: in testa il **nodo vecchio**, poi i **nodi legacy e topic**, il **buffer** e
le foglie sciolte/aperte. Quando scatta la soglia configurata (tre nodi per default), il nodo vecchio può assorbire i due
nodi idonei più vecchi, di qualunque natura, se passano anche i limiti di dimensione e risparmio. Le descrizioni dei topic assorbiti vengono
conservate parola per parola; la sintesi del pozzo è un'istantanea congelata, mentre le aggiunte
successive restano visibili nel topic. Le sintesi del pozzo superate si riaprono con `cwl_open`
usando id come `<pit id>.s1`.

**`cwl_recall`** / **`cwl_recall_episode`** — cercano testo compresso per parola chiave, oppure
recuperano un episodio evictato per nome, direttamente dal transcript.

### Installazione

Pi carica l'estensione da `index.ts` nella root del repository. Collega o copia l'estensione nella
directory delle estensioni di Pi:

```
~/.pi/agent/extensions/pi-cwl  ->  <questo repository>
```

### Configurazione

Opzionale, in `~/.pi/cwl/config.json`:

```json
{
  "tokenBudget": 80000,
  "thresholdRatio": 0.85,
  "levels": {
    "stripReasoning": true,
    "stripBulkOutput": true,
    "stripIntermediate": true,
    "removeEpisode": true
  },
  "protectedTurns": 4,
  "showWidget": true,
  "debug": false
}
```

- `tokenBudget` — budget di token attivi. `80000` è circa il 30% di una finestra di contesto da
  256k. Alzalo per finestre più grandi.
- `thresholdRatio` — l'eviction scatta a `tokenBudget × thresholdRatio`.
- `protectedTurns` — quanti dei confini di turno più recenti sono inviolabili (default `4`). Ogni
  confine porta con sé anche tutto ciò che lo segue, risvegli e carte di memoria compresi: è quindi
  la manopola che decide quanto contesto si riesce davvero a comprimere.
- `looseLeaves` — quante foglie restano sciolte, cioè visibili per intero, prima che una venga
  assorbita in un nodo (default `5`).
- `nodeCapacity` — quante foglie tiene un nodo prima di essere pieno (default `30`).
- `mergeNodesAt` — quanti nodi giovani fanno scattare l'accorpamento nel nodo vecchio (default `3`).
- `mergeMinRatio` — quante volte la sintesi liberata da un accorpamento deve valere piu' di quella che scrive (default `1.8`). E' il guard che passano sia un accorpamento sia un topic: la soglia di un topic e' `rapporto × 3.600` caratteri, mai meno di `mergeMinChars`, quindi con i default un topic deve liberare almeno `6.480` caratteri. Alzarlo rende i topic piu' rari e gli accorpamenti piu' difficili; abbassarlo fa l'opposto su entrambi.
- `mergeMinChars` — il minimo assoluto, in caratteri: sotto quella soglia un accorpamento viene rifiutato comunque (default `6000`).
- `topicInviteAt` — quante foglie puo' tenere il buffer prima che l'agente sia invitato ad aprire un topic con `cwl_group` (default `18`). Il buffer e' l'ULTIMO nodo e l'unico le cui foglie si possono ancora spostare: una foglia entrata in un nodo non si muove piu', quindi questo invito e' l'ultimo momento in cui un topic puo' nascere. Scatta solo quando il gruppo passerebbe il guard di dimensione, e porta con se' tutti i numeri per agire — l'id del nodo, quante foglie tiene, i caratteri di micro che portano, e i caratteri che il guard richiede.
- `showWidget` — disegna la forma dell'indice (pozzo / giovani / topic / sciolte / in attesa / testa) in una riga della TUI sotto l'editor. Una widget e' UI: non entra mai nel contesto e non costa un token (default `true`).

- `gate` — abilita la richiesta contestuale di comprimere materiale più vecchio quando il budget resta superato e l'eviction deterministica non ha altro da rimuovere (default `true`).

Le chiavi mancanti ricadono sui valori di default, quindi un file parziale è valido.

### Verifiche di sviluppo

Esegui `bash check.sh` per collegare le dipendenze Pi ed eseguire l'intera suite di test. Il typecheck parte solo se `tsc` è disponibile; verifica la riga `[check.sh] typecheck ran:` nell'output. Se non è presente, il typecheck è stato saltato.

### Attribuzione

Questa è un'**implementazione originale e indipendente**, scritta in TypeScript per l'ecosistema Pi
Agent. Non contiene codice copiato, forkato o adattato da alcun repository di terze parti.

- **Paper:** "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents"
  — Andrew Semenov, Svyatoslav Dorofeev (2026), [arXiv:2606.11213](https://arxiv.org/abs/2606.11213)
- Il concetto di CWL origina dagli autori del paper; questa implementazione è nostra.
- Vedi `CITATION.cff` per la citazione formale e `LICENSE` per i termini.
