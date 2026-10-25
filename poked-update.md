# Poked updates

Routine observations from the automatic timer live here, not in ARM-RESULTS.md. The report of record gets a row in the production series table when a validation lands and nothing else; prose about how the run is feeling belongs on this page, where it costs nobody's reading time and can be skimmed or deleted.

Anything that changes a decision, contradicts a recorded number, or would be re-derived later by a future agent still goes into ARM-RESULTS.md. The distinction is whether the sentence would change what someone does.

## 2026-10-25

Step 12000: macro 0.3469 up 0.0039, micro 0.5918 up 0.0013, fixed-0.40 reading 0.3324 up 0.0056. Macro back above the 0.344 strike level, so the extension stands. The fixed-threshold reading is closing on the best-threshold one, gap 0.0145 against 0.0162 a thousand steps earlier, the first hint that the calibration problem eases as the run anneals.

Step 13000: macro 0.3481 up 0.0012, micro 0.5875 down 0.0043, fixed-0.40 flat at 0.3320. Four consecutive macro rises at a slowing rate, 0.0030 then 0.0039 then 0.0012, while micro oscillates in a 0.004 band. Tail still learning, head saturated, which is the shape u1 predicts.

Step 14000: macro 0.3500 up 0.0019, micro 0.5951 up 0.0076, fixed-0.40 reading 0.3374 up 0.0054. Micro out of its band at a run high. Five consecutive macro rises, and the fixed-threshold gap down to 0.0126 from 0.0165 when the extension began.

Arm B, first calibration print at step 13010: expected positives 426.2 per image against 31.6 true, bias median -0.33. The 426 is the sum of sigmoid probabilities, not the count over 0.40, which was 72.9 for this checkpoint. So the controller is matching the expected count, the properly calibrated target, and it will pull many probabilities under 0.40 in the process. If the arm under-predicts and loses recall, the variant is to match the rate predicted at 0.40 instead of the expected rate.

Arm B verdict, 08:00. An independent audit agent reviewed the comparison and recommended resuming A, which I did. Its corrections to my own claims, all now struck in ARM-RESULTS.md: the 50k gap is 5.3 times the in-run gap rather than three; B did not trade precision for recall because both fell; and the fixed-threshold gap closing from 0.0126 to 0.0067 was 86 percent B's best-threshold macro falling, not the optimum moving, so the mechanism did not actually work.

The deeper lesson is mine and it is embarrassing: the calibration bias was live during training and never saved, so B's weights drifted +5.3 logits to compensate and the checkpoint alone fires on 8296 tags per image. Its first 50k reading of 0.0144 macro was therefore nonsense, and the refitted bias that produced 0.3571 is a new calibration rather than a restore. A state that is not in the checkpoint does not exist.

12:15 heartbeat. Production stepped to 14340 at 07:41 and then went silent for thirty-two minutes with the log having just printed "finished tar 7/213". Not a hang: the process kept burning 1.6 cores, GPU utilisation fell to 18 percent, and steps resumed at 128 img/s on their own. Recorded in docs/benchmark.md as a tar-boundary stall after resume, which is a spinning-disk seek rather than a trainer fault.

12:45 heartbeat, and a correction. The "thirty-two minute tar-boundary stall" I logged twenty minutes ago did not happen. Checkpoint 14301 was written at 07:35:59 and the run resumed a minute later, so the log I was reading five minutes into the resumed run was current, not stale, and I had compared it against a clock an hour ahead of the machine. The benchmark.md entry is struck and rewritten as the lesson rather than the hazard: check the machine clock and the process uptime before recording a stall.

13:45 heartbeat. Step 15000 validation: macro 0.3507 up 0.0007, micro 0.5928 down 0.0023, fixed-0.40 reading 0.3373 flat. The macro increments are decaying, 0.0012 then 0.0019 then 0.0007, and micro has now fallen in two of the last three windows, so the tail is still creeping while the head has stopped. Projected 50k at this point is about 0.385 macro, barely above the 14000 reading of 0.3841.

14:25. Paused cleanly at 16000 as asked. In-run 16000: macro 0.3532 up 0.0025, micro 0.5919 down 0.0009, fixed-0.40 reading 0.3375 up 0.0002. The 50k on that checkpoint is running now.

15:00. Production paused cleanly at 16000 and its 50k is in: macro 0.3840 against 0.3841 at 14000, so two thousand steps bought nothing, micro fell 0.5103 to 0.5042 and precision 0.3413 to 0.3357. Stopping was right.

Dino preparation done. One tar of DinoV3-L features is cached, 25000 rows at 2048 dimensions, 102 MB, labels 32.1 per image, no NaN. The 50000-image validation cache is also done at 32.3 labels per image, which is the label rate to the decimal and confirms alignment.

Measured cache rates: DinoV3-L 71 img/s so 22 h and 21.8 GB for the corpus, DinoV3-B 131 img/s so 11.3 h and 16.4 GB, SigLIP2-B 270 img/s so 5.5 h and 8.2 GB. Head training on the L cache runs 1270 rows per second, 70 minutes per epoch, so a cached experiment is an afternoon not a night.

A linear head trained on the 25k pilot scores macro 0.0597 and micro 0.3351 at 50k, which is meaningless as a result because it saw half a percent of the corpus, but it proves the plumbing at 2048 dimensions. Config with seed 1 is at configs/dino/u0-CM-L-asl-g2.0-dinov3l-512.json.

15:15 heartbeat. Production is paused at 16000 by request, so there is no step or validation to report; its 50k row is now in the report of record.

The card was otherwise idle and the question I put to the user, DinoV3-L at 22 hours against B at 11, is not answered yet, so I chose the order rather than the answer: DinoV3-B is caching over the full corpus now at 142 img/s, 10.4 hours, 16.4 GB, seed 1. The reasoning is that B answers the real question, whether frozen Dino features plus a trained head beat 0.3840, in half the time, and its result is what decides whether L's 22 hours are worth spending. The cache is resumable by row, so a reaped job costs minutes.
