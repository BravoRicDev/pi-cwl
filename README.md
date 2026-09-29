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
  "showWidget": true,
  "debug": false
}
```

- `tokenBudget` — active token budget. `80000` is roughly 30% of a 256k context window. Raise it for
  larger windows.
- `thresholdRatio` — eviction triggers at `tokenBudget × thresholdRatio`.

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
  "showWidget": true,
  "debug": false
}
```

- `tokenBudget` — budget di token attivi. `80000` è circa il 30% di una finestra di contesto da
  256k. Alzalo per finestre più grandi.
- `thresholdRatio` — l'eviction scatta a `tokenBudget × thresholdRatio`.

Le chiavi mancanti ricadono sui valori di default, quindi un file parziale è valido.

### Attribuzione

Questa è un'**implementazione originale e indipendente**, scritta in TypeScript per l'ecosistema Pi
Agent. Non contiene codice copiato, forkato o adattato da alcun repository di terze parti.

- **Paper:** "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents"
  — Andrew Semenov, Svyatoslav Dorofeev (2026), [arXiv:2606.11213](https://arxiv.org/abs/2606.11213)
- Il concetto di CWL origina dagli autori del paper; questa implementazione è nostra.
- Vedi `CITATION.cff` per la citazione formale e `LICENSE` per i termini.
