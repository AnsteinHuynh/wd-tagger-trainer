PAUSED ON PURPOSE 2026-10-23 03:10 local, by the user's instruction, not by a crash.

Step 15373 of 221739. That is 0.069 of the two-pass schedule, 738,000 images, 2 hours 4 minutes of training.

Why: to bracket `lr_unfreeze` 3e-5 and to price the fully frozen backbone (`unfreeze_blocks: 0`) on SigLIP2-B at 600000 images, before spending another 22 hours on the first setting ever inherited rather than measured. The three arms are `20142533` (lr 1e-4), `c75d34d4` (lr 1e-5) and `b6e293f0` (u0). This run is the fourth leg of that comparison and its whole point is that the current setting might be the good one, so it must not be deleted or overwritten.

What is on disk: `ckpt_step005000.pt`, `ckpt_step010000.pt`, `ckpt_step015000.pt`, `ckpt_step015373.pt`, and `best_val.pt` from step 10000 with macro@0.40 0.1775 and micro 0.5164. The `.PAUSE` file is still here and must stay until the resume actually starts, because its presence is what makes the relaunch exit immediately if anything else opens this directory.

To resume, delete nothing and run exactly this from the workspace root:

```
python -m wdtag.train --config configs\main\01_u1-C-L-asl-g2.0-siglip2b-main.json --select-metric macro040 --val-every-steps 10000 --ckpt-every-steps 5000
```

It resumes from `ckpt_step015373.pt` at step 15373 with the tar cursor read back from the checkpoint, so no image is shown twice across the seam. If the ablation says a different `lr_unfreeze` or `unfreeze_blocks: 0` is better, then this run is either abandoned as a wrong recipe or, if the ablation lands inside the band, resumed as is: that is the reason it was paused instead of killed.

Delete this file and the `.PAUSE` only when the run has actually restarted.
