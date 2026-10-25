"""Write the prune plan from the inventory; deletes nothing."""
import json

rows = json.load(open("scratch/sophie/prune_inventory.json"))
LIVE = {"459e2a02"}
rows = [r for r in rows if r["run"] not in LIVE]
rows.sort(key=lambda r: -r["gb"])
tot_gb = sum(r["gb"] for r in rows)
tot_rm = sum(r["rm"] for r in rows)
live = "459e2a02"

out = [
    "# Checkpoint prune plan, 2026-10-24",
    "",
    "Nothing here has been deleted. This is the list, written before any file is touched, so the run ids, the step each best_val.pt sits at and the size of every removal are on the record first. The rule is the one agreed: every run keeps best_val.pt and its highest-numbered ckpt_step file, everything else in the checkpoint set goes. The live run " + live + " is excluded entirely, and so is any run holding a pause file.",
    "",
    f"{len(rows)} run directories hold checkpoints, {tot_rm} intermediate files would go, freeing {tot_gb:.1f} GB against the 207.4 GB free on C: right now. The 190.3 GB of checkpoints becomes about {190.3 - tot_gb:.1f} GB, while the 14.9 GB of best_val.pt files is untouched by all of it. Debug pictures, logs and configs are all kept.",
    "",
    "The step column is read out of each best_val.pt, which is the only place that number is stored. It is the number that would have to be looked up before any of these checkpoints could be compared again, which is why it is written down here rather than left inside a file nobody loads twice.",
    "",
    "| run | best_val step | final checkpoint kept | files removed | GB freed |",
    "| --- | --- | --- | --- | --- |",
]
for r in rows:
    best = str(r["best_step"]) if r["best_step"] is not None else "no best_val"
    out.append(f"| {r['run']} | {best} | {r['final'] or 'none'} | {r['rm']} | {r['gb']:.2f} |")
out += [
    "",
    f"Total {tot_rm} files, {tot_gb:.1f} GB. Say the word and I run it one directory at a time, by hand, checking the free space after each batch rather than in one recursive sweep.",
    "",
]
# second list: run folders that never reached a checkpoint
empty = json.load(open("scratch/sophie/prune_empty.json"))
KEEP = {"_baseline": "holds the convnextv2_huge probability caches, which are the bar reference and are not regenerable for free"}
out += [
    "",
    "## Run folders with no checkpoint at all",
    "",
    "These never reached a checkpoint, so there is no model in them; what is left is a config, sometimes a hundred debug pictures, and in two cases something else that has to be looked at before it goes. They total " + f"{sum(e['gb'] for e in empty):.2f}" + " GB.",
    "",
    "| folder | files | MB | debug pictures | config | note |",
    "| --- | --- | --- | --- | --- | --- |",
]
for e in sorted(empty, key=lambda y: -y["gb"]):
    note = KEEP.get(e["run"], "")
    if e["run"] == "_probe":
        note = "stale frozen-feature caches from an abandoned probe; regenerable in minutes, keep unless you say otherwise"
    out.append(f"| {e['run']} | {e['files']} | {e['gb']*1024:.0f} | {e['pngs']} | {'yes' if e['cfg'] else 'no'} | {note} |")
out += [
    "",
    "The two named folders you pointed at, ca262bf3 and d5939d0d, are in this list: the first has a config and a hundred debug pictures, the second has nothing but a config.",
    "",
]
open("runs/PRUNE-2026-10-24.md", "w", encoding="utf-8").write("\n".join(out))
print("wrote runs/PRUNE-2026-10-24.md with", len(rows), "runs,", tot_rm, "files,", round(tot_gb, 1), "GB")


