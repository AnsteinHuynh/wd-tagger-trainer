# Current assistants

The tree of assistants alive on this project, who summoned them, what each one owns, and how it reports. One line per assistant, kept current: add on summon, amend on report, move to Retired when it is done. Sophie is the parent of everything in this session.

Sophie, parent, this session. Owns the card, `wdtag/`, the configs, `ARM-RESULTS.md` and every decision that spends GPU time.

committer-1, child of Sophie, summoned 2026-10-23 19:08. Owns one deliverable: a single clean commit of the working tree with a verified message. No push, no amend, nothing under `runs/` disturbed while `8854c266` is training. Reports the hash, the stat and anything it excluded.

## Retired

compacter-1, child of Sophie, summoned 2026-10-23 18:45, retired 19:05. Delivered `scratch/sophie/ARM-RESULTS.compact.md`, the compaction of `ARM-RESULTS.md` from 49792 to 22930 tokens, a 53.9% cut. Independently audited by Sophie with `scratch/sophie/audit_compact.py`: the same 45 numeric strings lost and all of them clock times, job ids, step markers of dropped check-ins or values present in another form; 42 of 43 sha8 kept, the missing one a dead default-hash mention; zero bold, zero list markers, zero thousands separators, zero period-terminated prose lines. Swapped into `ARM-RESULTS.md` at 19:06, original kept at `scratch/sophie/archive/ARM-RESULTS.precompact-2026-10-23.md`.
