# Pause, resume, and what "bit identical" can actually mean here

Everything below was measured on 2026-10-23 with `scratch/sophie/test_stream_cursor.py`, `scratch/sophie/test_step_determinism.py` and the real trainer, on a base tower at batch 8 over 2400 stream images, and it exists because a resume that silently repeats or drops images would poison every number in `ARM-RESULTS.md` without ever showing up in a log.

## The short version

The pause and resume paths are exact. Pausing at step 41 and resuming produces weights and optimiser state that are bit identical to an uninterrupted run of the same length, all 288 adapter tensors and all 80 optimiser states, and the same holds for a clean checkpoint stop and for cache-mode (epoch) resume.

But an ordinary production run cannot be compared that way, because the training step itself is not reproducible on this platform: two identical runs with the same seed and the same data diverge in the gradients from the very first step. That is flash attention's backward pass, which accumulates with atomics. So the honest claim is not "a resume is bit identical to no pause", it is "a resume is bit identical to no pause once the platform is made deterministic, and the platform's own run-to-run divergence is far larger than anything a pause adds".

## Evidence

Stream cursor, the thing that can silently replay or skip images: two independent `TarStream` readers over the same `(seed, pass_index)`, one from zero and one skipping 100 steps worth of samples, produce 12 out of 12 identical batches, byte for byte, images and labels and source stems. The skipped reader's first batch is the fresh reader's batch 101 and its `consumed` counter reports 808, which is 101 batches of 8.

Determinism floor, without any pause involved, at the first gradient: model initialisation hashes identically across processes, the first batch hashes identically, and the loss of that batch is identical to twelve decimal places. The gradient checksum is not. Turning on `--deterministic`, which disables flash and memory-efficient attention in favour of the math backend and turns off cudnn autotune, makes two separate processes produce 288 of 288 identical adapter tensors after 60 steps, worst absolute difference zero.

Resume exactness, with `--deterministic`: pause at step 41 by touching `.PAUSE`, resume, run to 120, compare with an uninterrupted 120-step control. Adapter tensors 288 of 288 identical, optimiser states 80 of 80 identical, the `best` record identical, and the logged loss at step 120 identical in the fourth decimal. Clean stop at 60 by `--max-steps` then resume to 120, 288 of 288. Cache mode with epochs, stop at 30 then resume to 60, 288 of 288.

What a pause costs in production: nothing extra. The divergence a resumed process shows against an uninterrupted one is the same size as the divergence between two uninterrupted runs, because both come from the same source, the non-deterministic attention backward. The cost of forcing determinism is real and is why it is not the default: 23 img/s against 55 img/s at batch 8 on the base tower, about 2.4x, and it would be worse at the sequence lengths the real runs use.

## Fixes made while verifying

Checkpoints are now written to `path.tmp` and then `os.replace`d into place. Before this, a job reaped during a 250 to 300 MB `torch.save` left a truncated file, and because resume takes the highest step number it would have taken the corrupt one and refused to start, with three good checkpoints sitting beside it.

Resume now walks checkpoint files newest to oldest and falls back on the first unreadable one with a warning, instead of dying. Tested by truncating `ckpt_step000120.pt` to half its length: the run warned, resumed from step 60 and finished.

A `.PAUSE` file present at launch no longer writes a `ckpt_step000000.pt` of 250 to 300 MB on every accidental relaunch. It exits immediately with a message and leaves the run directory untouched. This matters because the paused runs keep their `.PAUSE` file deliberately, `runs/3da79e14` and `runs/f63f6f54` both do, as the guard against a mistaken relaunch.

## What to do when a resume must be exact

Add `--deterministic` to both the interrupted leg and the resumed leg. Without it the comparison is meaningless, not because resume is wrong but because the reference itself moves. Never use it for a real training run: it costs more than half the throughput and the noise it removes is not a bias.

## Traps that remain

A `.PAUSE` at the workspace root stops every run, not just one, because that is what the last-resort check is for.

Resume picks the highest step number, so deleting the newest checkpoint deliberately replays those steps from the older state, which is correct rather than a bug, and costs whatever those steps cost.

`--init-from` is not a resume: it loads adapter and head weights only, with a fresh optimiser, a fresh schedule and a new run directory. A warm-started run is a new experiment even when its config is otherwise identical.

## Gradient accumulation

`grad_accum` in the config, default 1 and dropped from the hashed blob at that value, holds several micro-batches before one optimiser step, so a run can spend more images per update than fit in VRAM. `total_steps` divides by batch times accum, the resume skip multiplies by it, and the streamer is told how many batches the consumer keeps alive.

It is exact rather than approximate here because our ASL reduces with an elementwise mean over the batch and the mean of equal-sized micro-batch means is the whole-batch mean. Measured on the same twelve images: one batch of 12 against three of 4 agrees to 1.2e-5 relative on the loss and 0.999999 cosine on the gradients, with an rms difference of 1.4e-3 from bf16 kernel tiling. The BALMS reference loss, which normalises by each tag's positives inside the batch, would not have this property: a rare tag's positive gradient would be divided by the accumulation factor, biasing exactly the labels we care about.

One trap, found by testing rather than by reading. The pinned ring the streamer hands batches from was sized for a consumer holding a single batch. With accumulation the consumer holds the whole group, the ring stops covering what is live, and the producer overwrites batches the trainer has not consumed yet. Two runs of the same config then agree for fifty steps and drift apart at a random step, on all trainable tensors at once, which reads like a numerical problem and is not. `TarStream(hold=...)` sizes the ring for it; the trainer passes `hold=grad_accum`.

Verification after that fix, all under `--deterministic` with dropout off. Two independent runs of 3 by 4: bit-identical at steps 20, 40, 60, 80, 100 and at 300. Pause at step 151 and resume: bit-identical to the uninterrupted control at 300. Edge cases: a stream length that is not a whole number of accumulated steps finishes and logs the shortfall; `.PAUSE` before the first step exits with zero checkpoints written; a truncated newest checkpoint falls back to the previous one and continues.

## Stopping at a chosen step, or at the next validation

`.PAUSE` stops at the next step boundary. Two suffixed forms are for stopping at a place you will want to read rather than as soon as possible.

`.PAUSE-6300` stops after step 6300 has finished and its checkpoint has been written, so the file exists at exactly the step you named. Write `.PAUSE-6300` while the run is at 6040 and it steps to 6300, saves, prints `[pause] step target 6300 reached`, and exits.

`.PAUSE-VALID` stops after the next validation rather than before it, so the numbers for that step are computed and printed and `best_val.pt` is updated before the process ends. This is the form to use when you want a scored checkpoint rather than a quiet one.

Both are re-read every step, so writing one into the run directory while training runs is enough; nothing needs restarting. Both delete themselves at the moment they fire, so a resume continues past the point instead of pausing again forever.

Four rules that settle the awkward cases, and the startup line prints which of them applies.

A numbered target behind the current step, like `.PAUSE-6000` written at 6040, pauses at the next step boundary rather than being ignored.

A numbered target that lands on a validation step runs the validation first, then pauses, so both the checkpoint and its numbers exist.

`.PAUSE-VALID` with `--val-every-steps 0` cannot wait for a validation that will never come, so it warns at startup and falls back to pausing at the next step boundary.

If both a numbered target and `.PAUSE-VALID` are present, whichever fires first wins, every file that fired is removed, and the startup line says so.
