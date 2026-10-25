# Writing style, and how to spend fewer tokens on it

Every markdown file in this repo is read by an assistant sooner or later, and every token spent on decoration is a token not spent on evidence. These rules exist because the user asked for them three separate times, so treat a violation as a defect to fix while you are in the file

This document obeys itself. Read the shape of it, not just the words: each paragraph below is a single line

## The five rules

No emphasis. No `**bold**`, no `*italic*`, no `__underline__`. Emphasis costs a token or two per span, buys no meaning, and reads as shouting. If a sentence matters, say it plainly

No list markers. A bullet becomes a plain line with no leading `*`, `-` or `+`. Keep the indentation when the nesting carries meaning, drop the marker

No terminal periods. No line in this repo ends with a full stop, whether it is a bullet that became a plain line, a lead-in sentence, or a whole paragraph. Punctuation inside the line stays, including the period before a dash or before a parenthetical. A period at the end of a line is a token that tells the reader nothing the line break did not already say

No thousands separators in numbers. Write 1253 images, 12476 labels, 5321713 training images, 638.8 MB, 30.6 GB. A separator costs a token each time and reads as a list separator; the bad forms are the ones written with a comma between the thousands and the hundreds. Decimals and ranges keep their punctuation: 0.4791, 0.40-0.45, 1e-7

No hard wrapping. One paragraph is one line, however long it gets. Do not break at 80 or 100 columns, do not split a sentence across lines, and do not reflow text you were not asked to change. Line breaks carry meaning only in fenced code, tables, and column-aligned indexes

Also: delete "Back to index" links. They are navigation for a file nobody reads linearly

## Compress lossily, keep facts exactly

The goal is fewer words carrying the same evidence, not fewer facts. Compress by deleting whole clauses that do no work, never by rounding a number or dropping a caveat

Delete at the sentence level: restated conclusions, narrative of how a finding was reached once the finding is stated, hedging ("it is worth noting", "in general terms"), lead-in sentences that announce a table instead of adding to it, duplicated derivations of the same figure across sections, and the history of an earlier wrong value unless the wrong value is itself a trap worth avoiding

Keep, always: every measured number with its condition (batch, resolution, config, run id), citation and equation numbers plus the warning when a source is cited wrongly elsewhere, named traps and the failure they caused in one line, the WHY behind a non-obvious constant even when it costs 20 tokens, provenance marks such as measured, inferred, hypothesis, and formulas copied rather than paraphrased

The test is not "is it shorter" but "could a reader still prevent the bug this paragraph exists to prevent".

## Worked example, markers and restatement

Before, 68 tokens:

```markdown
### 3.1 Recommended: uncompressed tar shards

What we use, and what I would reach for again.

* **Sequential by construction** — concatenated members, one forward pass, head never reverses.
* **No index** — any byte offset, headers parsed forward; resume trivial and exact (§5.3).
```

After, 62 tokens, same two facts:

```markdown
### 3.1 Recommended: uncompressed tar shards

What we use, and what I would reach for again

Sequential by construction — concatenated members, one forward pass, head never reverses
No index — any byte offset, headers parsed forward; resume trivial and exact (§5.3)
```

That is only -9%, and it is the honest size of a pure marker strip. The markup is not where the tokens are

## Worked example, where the tokens actually are

Restatement and hedging cost far more than markers. Before, 47 tokens:

```text
It is worth noting that throughput is governed by the number of unfrozen blocks rather than by the
compiler. In other words, the compiler does not determine throughput here; the unfrozen block count
does. This has been observed repeatedly and is the main thing to keep in mind.
```

After, 12 tokens:

```text
Throughput is governed by unfrozen block count, not by the compiler
```

## Measure, do not estimate

Count with the project tokenizer, never by eyeballing characters:

```powershell
cd C:\Users\kerok\apps\wd-tagger-trainer
C:\Python314\python.exe -c "from tokenizers import Tokenizer;t=Tokenizer.from_file(r'I:\Models\tokenizers\DeepSeek-V4.1-Flash\tokenizer.json');s=open(r'docs/benchmark.md',encoding='utf-8').read();print(len(t.encode(s).ids),'tokens',len(s),'chars')"
```

Report before and after with the same command, on the same file, in the same session. A percentage you did not measure is a claim, not a result

## Know the floor before promising a number

Prose has a hard floor on these files because tables and code are protected content. Measured on this repo: oft-saturation.md is 51.7% tables and code by token count, so no amount of prose editing reaches a 40% cut. hdd-image-pipeline.md is 25.7% protected, and mechanically deleting every parenthetical it contains tops out at -19.4%. hyperparameter-scaling.md is 46% fixed mass, and 1204 of its remaining prose tokens ARE the numeric strings, so deleting all connective text still leaves 3327 tokens

A realistic target is 10-25% per file. If you cannot reach your target without cutting evidence, stop and say so with the arithmetic. Never cut a fact to hit a number, and never claim a compression you did not measure

## Check before you save

No `**`, no `*emphasis*`, no `__underline__` outside code spans. No line starting with `*`, `-` or `+`. No comma between thousands and hundreds anywhere, including table cells and inline code samples. No wrapped paragraphs: every prose paragraph and list item is exactly one line. Tables, fenced code, headings and column-aligned indexes are byte-identical to before unless the task was to change them. Numbers audited: extract every distinct numeric string before and after and compare the sets
