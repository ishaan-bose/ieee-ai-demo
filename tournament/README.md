# tournament

Separate CLI, run manually by the owner over SSH after the event (SPEC section 10).
The API server never imports or starts it. Built in phase 10.

- `engine/`: python-chess legal moves + deterministic alpha-beta
- `run.py`: Swiss rounds CLI
- `ratings.py`: Bradley-Terry / Elo
- `tests/`
