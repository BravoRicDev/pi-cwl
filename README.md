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
- **User turns are inviolable.** User and system messages are never evicted.
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

**`delimiter`** — marks episode boundaries.

| Parameter | Type | Notes |
|---|---|---|
| `action` | `"start"` \| `"end"` | Required |
| `name` | `string` | Required on `start`. Must be unique |
| `type` | `"expl"` \| `"act"` | Required on `start` |
| `dependencies` | `string[]` | Names of closed `expl` episodes an `act` depends on |
| `description` | `string` | Required when closing an `expl`: what you learned |

**`cwl_status`** — current token budget, measured context size, episode counts, evictions performed
and tokens saved.

**`cwl_compress`** / **`cwl_compress_range`** — compress an explicit range of the conversation into
a summary. `cwl_compress_range` picks the oldest eligible interval for you, and accepts an optional
`micro` (the label that will represent the leaf in the index).

**`cwl_open`** — reopen a leaf: returns the full body, never truncated, and declares its size first.

**`cwl_micro`** — replace a leaf body with a short label (ceiling: 1,200 characters, ~300 tokens);
an empty `text` gives the body back.

**`cwl_old`** — fold the oldest young nodes into the old node: the summary of summaries.

**`cwl_group`** — group leaves into a **topic node**, which is born collapsed: the `name` and the
`description` you write stand for its leaves in the index from that moment on, so the description
must already cover the future use of the topic. Only the leaves of the buffer — the last node, the
one attached to the leaves still open — can be grouped or moved, and a topic is never the first
node. Adding leaves to an existing topic (`node`, without `description`) is free: the labels leave
the head, the description stays. The description is immutable while the topic is OUTSIDE the old
node; once the topic is inside it, it can be rewritten, because there it no longer touches the
context.

The index is therefore NESTED: the **old node** at the head, then **legacy nodes** and **topic
nodes**, then the **buffer** (which is never pruned, even at zero leaves), then the leaves still
open. With three nodes in front of it, the old node absorbs two, of any kind, and the descriptions
of the topics it swallows are glued to its synthesis word for word — they are not rewritten, and
the tip's synthesis is a frozen snapshot: material added to a topic afterwards is described by the
topic, not by the pit, and both are readable with `cwl_open`.

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
- `mergeMinRatio` — how many times the synthesis a merge frees must outweigh the one it writes (default `3`).
- `mergeMinChars` — the absolute floor, in characters: below it a merge is refused whatever the ratio says (default `6000`).
- `showWidget` — draws the index shape (pit / young / topics / loose / waiting / head) in one TUI line below the editor. A widget is UI: it never enters the context and costs no tokens (default `true`).

Missing keys fall back to defaults, so a partial file is valid.

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
- **I turni dell'utente sono inviolabili.** I messaggi utente e di sistema non vengono mai evictati.
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

**`delimiter`** — segna i confini di un episodio.

| Parametro | Tipo | Note |
|---|---|---|
| `action` | `"start"` \| `"end"` | Obbligatorio |
| `name` | `string` | Obbligatorio su `start`. Deve essere univoco |
| `type` | `"expl"` \| `"act"` | Obbligatorio su `start` |
| `dependencies` | `string[]` | Nomi degli episodi `expl` chiusi da cui un `act` dipende |
| `description` | `string` | Obbligatorio alla chiusura di un `expl`: cosa hai imparato |

**`cwl_status`** — budget corrente, dimensione misurata del contesto, conteggio episodi, eviction
eseguite e token risparmiati.

**`cwl_compress`** / **`cwl_compress_range`** — comprimono un intervallo esplicito della
conversazione in un riassunto. `cwl_compress_range` sceglie per te l'intervallo più vecchio
disponibile e accetta un `micro` opzionale: l'etichetta che rappresenterà la foglia nell'indice.

**`cwl_open`** — riapre una foglia: restituisce il corpo intero, mai troncato, e ne dichiara prima la
dimensione.

**`cwl_micro`** — sostituisce il corpo di una foglia con un'etichetta breve (tetto: 1.200
caratteri, ~300 token); un `text` vuoto restituisce il corpo.

**`cwl_old`** — accorpa i nodi giovani più vecchi nel nodo vecchio: il riassunto dei riassunti.

**`cwl_group`** — raggruppa le foglie in un **nodo topic**, che nasce già collassato: il `name` e la
`description` che scrivi stanno per le sue foglie nell'indice da quel momento, quindi la descrizione
deve coprire già l'uso futuro del topic. Si possono raggruppare o spostare solo le foglie del buffer
— l'ultimo nodo, quello attaccato alle foglie ancora aperte — e un topic non è mai il primo nodo.
Aggiungere foglie a un topic esistente (`node`, senza `description`) è gratis: le etichette escono
dalla testa, la descrizione resta. La descrizione è immutabile finché il topic è FUORI dal nodo
vecchio; una volta che il topic è dentro, si può riscrivere, perché lì non tocca più il contesto.

L'indice è quindi ANNIDATO: in testa il **nodo vecchio**, poi i **nodi legacy** e i **nodi topic**,
poi il **buffer** (che non viene mai potato, nemmeno a zero foglie), poi le foglie ancora aperte.
Con tre nodi davanti, il nodo vecchio ne assorbe due, di qualunque natura, e le descrizioni dei
topic che ingoia vengono incollate alla sua sintesi parola per parola — non vengono riscritte, e la
sintesi del pozzo è un'istantanea congelata: il materiale aggiunto dopo a un topic lo racconta il
topic, non il pozzo, e si leggono entrambi con `cwl_open`.

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
- `mergeMinRatio` — quante volte la sintesi liberata da un accorpamento deve valere piu' di quella che scrive (default `3`).
- `mergeMinChars` — il minimo assoluto, in caratteri: sotto quella soglia un accorpamento viene rifiutato comunque (default `6000`).
- `showWidget` — disegna la forma dell'indice (pozzo / giovani / topic / sciolte / in attesa / testa) in una riga della TUI sotto l'editor. Una widget e' UI: non entra mai nel contesto e non costa un token (default `true`).

Le chiavi mancanti ricadono sui valori di default, quindi un file parziale è valido.

### Attribuzione

Questa è un'**implementazione originale e indipendente**, scritta in TypeScript per l'ecosistema Pi
Agent. Non contiene codice copiato, forkato o adattato da alcun repository di terze parti.

- **Paper:** "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents"
  — Andrew Semenov, Svyatoslav Dorofeev (2026), [arXiv:2606.11213](https://arxiv.org/abs/2606.11213)
- Il concetto di CWL origina dagli autori del paper; questa implementazione è nostra.
- Vedi `CITATION.cff` per la citazione formale e `LICENSE` per i termini.
