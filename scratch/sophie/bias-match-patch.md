# Proposal: per-tag rate-matching bias, exact patch

Measured gain first, from `docs/calibration-at-fixed-threshold.md`: a running per-tag bias that matches each tag's predicted rate to its label rate reads 0.3781 macro on held-out images against 0.3686 for a temperature alone and 0.3465 uncalibrated. Online beat the offline grid search, so this is the version to ship.

Not applied yet, on purpose: the card was at 21.2 GB of 24.5 during production and the change touches both the loss path and the validation path, so it cannot be smoke tested without stopping the run. Apply it, then test it on a 400-step arm before trusting it.

## 1. config.py

Add after `loss_neg_relax_alpha`:

```python
    # Per-tag rate matching. Every `bias_match_every` steps, nudge each tag's logit bias so its
    # predicted positive rate over the accumulated batch approaches its label rate. Fixes the
    # over-prediction that costs macro at the fixed 0.40 threshold. Zero = off.
    bias_match_eta: float = 0.0
    bias_match_every: int = 50
    bias_match_clip: float = 6.0
```

## 2. train.py, after the loss is constructed

The bias must apply to the logits on every path: training, in-run validation, and the checkpointed score. The cleanest place is a closure around the forward call.

```python
    cal_bias = torch.zeros(model.n_labels, device=device, dtype=torch.float32)
    _raw_forward = model.__call__

    def _forward(x, _f=_raw_forward):
        return _f(x) + cal_bias if cfg.bias_match_eta > 0 else _f(x)
```

Then pass `_forward` wherever `model(...)` was called in the step loop, and to `evaluate(...)` as its `forward` argument.

## 3. train.py, after `opt.step()` inside the step loop

```python
    if cfg.bias_match_eta > 0 and step % cfg.bias_match_every == 0:
        with torch.no_grad():
            p = torch.sigmoid(_step_logits.float())
            pred = p.mean(0)
            want = y_batch.float().mean(0)
            cal_bias.add_((want - pred) * cfg.bias_match_eta * 8.0)
            cal_bias.clamp_(-cfg.bias_match_clip, cfg.bias_match_clip)
```

`_step_logits` and `y_batch` are the accumulated logits and targets already in scope for the loss; keep a reference to them when accumulating over micro-batches. If they are not kept, rely on the final micro-batch only and halve eta: the signal is a rate, not a per-image quantity.

## 4. print the ratio every thousand steps

```python
    if step % 1000 == 0:
        print(f"[cal] predicted/true labels per image "
              f"{float(p.sum(1).mean()):.1f}/{float(y_batch.sum(1).mean()):.1f}", flush=True)
```

That single line is the whole point of the change: a ratio near 1.0 means calibration is holding, a ratio near 2.0 means the run is losing macro to a scalar.

## 5. save it with the checkpoint

`cal_bias` must go into the checkpoint dict and be restored on resume, or a resumed run silently reverts to uncalibrated. Add it beside the optimizer state in `save_ckpt`, and in `load_state_dict`, guarding for its absence so old checkpoints still load.

## 6. the test that decides whether it ships

One arm, 400 steps, 150000 images at effective 384, with `bias_match_eta` 2.0 and `bias_match_every` 50, against the control's first 400 steps. The pass condition is the printed ratio falling from its uncalibrated value toward 1.0 without macro falling, and the failure mode to watch is a tag whose bias runs away to the clip because its label rate is noisy at 384 images per batch. If that happens, the fix is per-tag shrinkage by count, exactly as the offline version required for tags under 200 positives.

## Two facts checked against the code before proposing this

`model.n_labels` exists, model.py line 128, so the closure can size the bias without plumbing a new argument through. And `init_head_bias(priors)` at train.py line 594 already sets the head bias to logit(prior), which is BALMS's prior correction. So the head starts calibrated and the run still ends up predicting 72.9 tags per image against 32.3 true. The over-prediction is therefore not an initialisation mistake that someone forgot to fix, it is the loss walking the bias away from the prior over ten thousand steps, which is exactly what a running controller can hold back.

## Hard requirement added after the first attempt failed, 2026-10-25

The first implementation did not save `cal_bias` into the checkpoint, and the arm it produced is unusable without surgery: the weights drift about +5.3 logits to compensate for a bias that is applied during training, so the checkpoint alone fires on 8296 tags per image and scores macro 0.0144 where the arm actually read 0.3449 in-run. Reconstructing the bias by replaying the controller on fresh images is not a restore, it is a new calibration fit, and it saturated at the clip.

So section 5 above is not optional, it is the condition for the change being usable at all: `cal_bias` goes in the checkpoint dict, comes back on resume, and the validator applies it through `--cal-bias`. That flag now exists in `wdtag/validate.py`, and a run that trains with rate matching must emit its bias beside every checkpoint.

Second lesson from the same arm: with the clip at 6 the controller never reached the label rate, holding about 340 expected positives per image against 32.3 true. If rate matching is tried again, either the clip opens wider or eta is applied as a fraction of the needed shift, because a controller that saturates is a constant, not a controller.
