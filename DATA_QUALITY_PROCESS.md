# Data-Quality Process

How the data in this project is checked before analysis. The process is written so that the checking step can later be run by an automated agent, while every decision that changes the data stays with a person.

```
01 structural cleaning  →  02 automatic checks  →  human review  →  03 apply decisions  →  analysis
   (no judgement)          (finds, never fixes)    decisions.csv     (only what was approved)
```

## Roles

| Step | Who | Can change the data? |
|---|---|---|
| 01 Structural cleaning | code | Only fixes with one correct answer: layout, units, a mis-dated week confirmed by 7-day spacing, exact duplicates whose values match |
| 02 Automatic checks | code / agent | **No.** Writes `data/review/issues.csv` and adds new rows to `decisions.csv` |
| Review | **person** | Fills `decision`, `value`, `note`, `reviewer` in `decisions.csv` |
| 03 Apply decisions | code | Applies approved decisions only; marks output `provisional` while anything is pending |

## Checks (`dq/checks.py`)

| Check | Finds | Default severity |
|---|---|---|
| `missing_values`, `duplicates` | Empty cells, repeated keys | high |
| `week_start_day` | A week that does not start on Sunday | high |
| `missing_weeks` | Gaps in the weekly series | medium |
| `parts_vs_total` | Cities or sectors that do not add up to the national total | high |
| `name_mismatch` | A name in one source with no exact match in the other, with the closest candidates | medium |
| `outlier` | A week far from its neighbouring weeks (robust z-score > 3.5 on log values vs a 9-week rolling median) | low if a known event explains it, otherwise medium / high |
| `published_change_mismatch` | The source's own % change disagrees with the values | medium |

**Severity** — `high`: likely an error, blocks use · `medium`: needs a person · `low`: probably real, listed for completeness.

**Known events** used to explain unusual weeks: Ramadan, last 10 days of Ramadan, Eid al-Fitr, Hajj, Eid al-Adha (Hijri, Umm al-Qura), COVID curfew and reopening, school start, Saudi National Day, Founding Day, White Friday, year-end holidays. When a reviewer confirms a new explanation, add it to `GREGORIAN_EVENTS` or `DATE_RANGES` so it is recognised next time.

## Decision vocabulary

| Check | Decisions |
|---|---|
| `name_mismatch` | `map` (rename to `value`) · `separate` |
| `outlier` | `keep` · `exclude` (flags the week; nothing is deleted) |
| `missing_weeks` | `document` |

## Rules an agent running step 02 must follow

1. Never edit raw, interim or clean data. Only write to `data/review/`.
2. Never overwrite an existing decision. New issues are appended; issue ids are stable hashes of `check | table | key`, so a re-run finds the same issue under the same id.
3. Pre-fill only where the answer is mechanical (`low` outliers → `keep`, gaps → `document`). Leave every judgement call empty.
4. For each pending issue, give the evidence and a suggestion in plain words; rank by severity.
5. Report how many issues are pending. Analysis outputs stay marked `provisional` until that number is zero.
