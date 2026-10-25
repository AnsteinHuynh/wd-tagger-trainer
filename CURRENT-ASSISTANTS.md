# Current assistants

The tree of assistants alive on this project, who summoned them, what each one owns, and how it reports. One line per assistant, kept current: add on summon, amend on report, move to Retired when it is done. Sophie is the parent of everything in this session.

Sophie, parent, this session. Owns the card, `wdtag/`, the configs, `ARM-RESULTS.md` and every decision that spends GPU time.

committer-1, child of Sophie, summoned 2026-10-23 19:08, resumed 19:12. Owns git hygiene: the first commit of the working tree (`96a9b69`, 447 files) and now the ignore rules for scratch artifacts. No push, no amend, nothing under `runs/` disturbed while `8854c266` is training. Reports the hash, the stat and anything it excluded.

speed-auditor-1, child of Sophie, summoned 2026-10-23 23:05. Owns the throughput question: why the trainer runs far below what its own synthetic bench predicts, why batch 24 to 40 only moved L/384 from 59 to 68 img/s, and whether the once-per-step GPU dip is decode starvation or a host sync. Reads code and logs only, no GPU work while the card is busy, writes `scratch/sophie/speed-audit.md`, reports a ranked suspect list with milliseconds per step and one cheap test each. Retired 23:35 after delivering the audit; its findings became `docs/throughput-budget.md`.

gold-settings-1, child of Sophie, summoned 2026-10-23 23:45. Owns the question of what the user calls the gold-starred optimizer settings: find them in `speedy-sdxl-vae-trainer` or `dreamsim-dinov3` or the opt-report audits, quote them verbatim with file and line, say what evidence backed them, and judge whether they transfer to SinkSGD_adv on OFTv2 with ASL and a 12476-way head. Read only, no GPU, writes `scratch/sophie/gold-settings.md`. Retired 00:15 after applying the four switches to `configs/best.json` and `configs/full_corpus_5p3m.json` on the user's instruction; those two files still carry them and were not used for the production run, which is plain SinkSGD with fp32 state.

rung-displacement-1, child of Sophie, summoned 2026-10-24 00:35. Owns the measurement that separates a switch changing direction from a switch changing step length: the per-rung step displacement over the live u1 parameter inventory with weights restored between probes. Writes `scratch/sophie/rung-displacement.md` and the raw JSON beside it. Retired 01:18. Its numbers are what moved the ladder to matched-throttle arms and what condemned spectral normalization on untagged OFT tensors.

## Retired

compacter-1, child of Sophie, summoned 2026-10-23 18:45, retired 19:05. Delivered `scratch/sophie/ARM-RESULTS.compact.md`, the compaction of `ARM-RESULTS.md` from 49792 to 22930 tokens, a 53.9% cut. Independently audited by Sophie with `scratch/sophie/audit_compact.py`: the same 45 numeric strings lost and all of them clock times, job ids, step markers of dropped check-ins or values present in another form; 42 of 43 sha8 kept, the missing one a dead default-hash mention; zero bold, zero list markers, zero thousands separators, zero period-terminated prose lines. Swapped into `ARM-RESULTS.md` at 19:06, original kept at `scratch/sophie/archive/ARM-RESULTS.precompact-2026-10-23.md`.
