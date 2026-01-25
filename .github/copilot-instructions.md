# Copilot instructions for Wyrmspan AI agents

These notes give an AI coding assistant the specific, actionable context needed to work productively in this repository.

**Big Picture**
- **Core modules**: [game_states.py](../game_states.py) (data model: `GameState`, `PlayerState`, constants, loads `data/*.json`), [game_logic.py](../game_logic.py) (game rules and state transitions), [playout_compare.py](../playout_compare.py) (simulation algos & runners), [read_game.py](../read_game.py) (replay saved sequences).
- **Data sources**: [data/dragon_cards.json](../data/dragon_cards.json), [data/cave_cards.json](../data/cave_cards.json), [data/guild_tiles.json](../data/guild_tiles.json), [data/objective_tiles.json](../data/objective_tiles.json), [data/automa_cards.json](../data/automa_cards.json) — card and tile definitions are indexed by integer IDs (dragon IDs 1..183, cave IDs 1..75).
- **Execution model**: Game progression is driven by `game_logic.get_next_state(game_state, choice_or_event)`; the engine sets `game_state.current_choice` (player choices) or `game_state.current_random_event` (random outcomes). Agents should update or choose actions by returning an index into `current_choice` or supplying an RNG outcome.

**Project-specific conventions & patterns**
 - Card IDs are integers and used as indices into the JSON-loaded lists (1-based IDs are used directly in many places). See [game_states.py](../game_states.py) for constants and JSON loads.
 - Reproducibility uses `RNGOrder` in [playout_compare.py](../playout_compare.py) to pre-seed decks and automa decisions — use this for deterministic simulations and for replaying saved RNG orders.
 - State copying: `GameState.make_copy()` and deep copies are used heavily before simulations; avoid mutating shared `GameState` instances when running parallel simulations.
 - Logging: modules use Python `logging`; `read_game.py` configures `read_game.log` for reproduction/debugging. Prefer adding logger.debug/info statements when diagnosing policies.

**Developer workflows & examples**
 - Replay a saved sequence (uses relative paths — run from project root):

```
python read_game.py --file saved_sequences/sequence_53125_1_20250806-182635_0.1089.json
```

 - Run a quick algorithm comparison interactively (small example):

```
python -c "from playout_compare import compare_algorithms; compare_algorithms(num_simulations=50)"
```

 - Replay + deterministic RNG: create a `SoloGameState`, then use `RNGOrder(game)` and `playout_compare.simulate_game_given_rng(...)` to run identical random-event sequences.

**Concurrency & performance notes**
 - `simulate_multiple_games` and `compare_algorithms` use `concurrent.futures.ProcessPoolExecutor` — simulations are CPU-bound and expect independent deep-copied states. Keep payloads picklable and avoid sending open file handles across processes.

**What to look for when changing logic or adding agents**
 - When adding a policy function, implement it to accept a `GameState` and return an index (see `get_sim_algo` in [playout_compare.py](../playout_compare.py)). Use `RNGOrder` if your policy depends on deterministic random draws.
 - Maintain compatibility with the event-driven loop: if you call `game_logic.get_next_state`, ensure you pass None or a numeric index consistent with `current_choice` semantics.
 - Tests & experiments are ad-hoc; prefer creating small scripts in `scripts/` or Jupyter notebooks (`simulate.ipynb`, `convert_to_json.ipynb`) for reproducibility rather than modifying top-level modules.

**Files to consult for examples**
 - Replay: [read_game.py](../read_game.py)
 - Simulations & policies: [playout_compare.py](../playout_compare.py)
 - Rules & conversions: [game_logic.py](../game_logic.py)
 - Data and types: [game_states.py](../game_states.py)
 - Saved experiment traces: [saved_sequences](../saved_sequences/) (JSON lists of chosen action indices/events)

If any of these sections lack detail you'd like, tell me which area (replay, simulation, data model, or concurrency) and I'll expand examples or add runnable scripts. 
