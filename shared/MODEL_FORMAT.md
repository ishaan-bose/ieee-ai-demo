# Saved model format (SPEC 8.4)

The ONLY interface between training and the tournament. One loader, used by both:
`backend/app/chess_net/model_io.py` (`load_model(dir, device)` -> `LoadedModel.score(boards, stm, castle, ep)`).
The position encoder is also one module for both: `backend/app/chess_net/encoder.py`.

```
models/<submission_id>/            (under STATE_DIR; the House Net lives in models/house-net/)
  weights.safetensors   fp32 state dict of app.chess_net.model.EvalMLP
  config.json           {"config": <resolved config>, "param_count", "tier", "search_depth_full_moves", "nickname", "model_name"}
  meta.json             stop_reason (budget|time|diverged|killed|error), flops_used, flops_budget, active_gpu_seconds,
                        samples_seen, steps, preemptions, val_loss, val_mse, train_loss, started_at, finished_at, device
  curves.json           list of {t (active s), step, samples_seen, flops, train_loss, val_loss, val_mse}
```

`config.json -> config` is the full resolved config (every knob of SPEC 8.3 plus the derived `widths`, `in_dim`,
`param_count`, `matmul_params`, `tier`, `search_depth_full_moves`). Fields that matter to the tournament:
`widths, activation, normalization, residual, output_head, input_extras, target_type, eval_squash_scale, mate_clip, perspective_flip`.

## Meaning of the network output
The net outputs one raw value `z`. `output_head` and `target_type` map it to the training-target space
(`encoder.head_to_target_space`); `perspective_flip` says whether that value is for the side to move (true) or for White (false).
`LoadedModel.score` converts everything to a **centipawn-like score for the side to move**, which is what the engine uses as a leaf value:
winprob targets use `K * logit(p)`, cp targets use `value * K`.

## Determinism
`score` evaluates in fixed-size fp32 chunks of 256 (padded), with TF32 off, so a position's score never depends on which other
positions share its batch.
