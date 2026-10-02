# backend/scripts

One-off scripts the owner runs on the server (SPEC section 3). Planned:

| Script | Phase | Purpose |
|---|---|---|
| `rasterize_quickdraw.py` | 2 | stroke JSONL -> `quickdraw/tensors/*.npy` (28x28 uint8) |
| `benchmark.py` | 3 | the section 11 benchmark report |
| `record_races.py` | 5 | record cached race streams for the frontend |
| `train_showcase.py`, `export_weights.py` | 7 | browser showcase models |
| `train_house_net.py` | 8 | the House Net |

Nothing here yet (phase 1).
