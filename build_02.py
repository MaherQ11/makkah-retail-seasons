import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Makkah Retail Sales — 02: Data-Quality Review

This notebook **finds** problems; it does not fix them. Each check in `dq/checks.py` returns a list of issues, ranked by severity, with the evidence and a suggested action.

The output is a review queue for a person:

| File | Who writes it | What it holds |
|---|---|---|
| `data/review/issues.csv` | this notebook | every issue found |
| `data/review/decisions.csv` | **a person** | `approve`, `reject` or `edit` for each issue, plus a note |

`03_apply_decisions.ipynb` then applies only what the person approved.

The checks are written as plain functions so they can later be run by an automated agent on new data.
""")

code("""
from pathlib import Path
import pandas as pd
from dq import checks

INTERIM, REVIEW = Path("data/interim"), Path("data/review")
REVIEW.mkdir(parents=True, exist_ok=True)

cities = pd.read_csv(INTERIM / "cities_weekly.csv", parse_dates=["starting_date"])
sectors = pd.read_csv(INTERIM / "sectors_weekly.csv", parse_dates=["starting_date"])
national = pd.read_csv(INTERIM / "national_weekly.csv", parse_dates=["starting_date"]).set_index("starting_date")
published = pd.read_csv(INTERIM / "published_changes.csv", parse_dates=["starting_date"])

cal = checks.calendar_flags(pd.concat([cities.starting_date, sectors.starting_date]))
cities = cities.merge(cal, on="starting_date", how="left")
sectors = sectors.merge(cal, on="starting_date", how="left").assign(covid_period=lambda d: d.starting_date < "2020-06-21")
issues = []
""")

md("## Check 1 — Missing values and duplicates")
code("""
issues += checks.check_missing_and_duplicates(cities, "cities", ["starting_date", "city"])
issues += checks.check_missing_and_duplicates(sectors, "sectors", ["starting_date", "sectors"])
len(issues)
""")

md("## Check 2 — Week regularity (every week starts on Sunday, 7 days apart)")
code("""
for city, g in cities.query("city == 'Makkah'").groupby("city"):
    issues += checks.check_week_regularity(g, "cities (Makkah)")
issues += checks.check_week_regularity(sectors, "sectors")
[i for i in issues if i["check"] in ("week_start_day", "missing_weeks")]
""")

md("## Check 3 — Parts add up to the total (CSV source)")
code("""
csv_cities = cities.query("source == 'kapsarc_csv'")
issues += checks.check_parts_sum_to_total(csv_cities, national, "cities", "starting_date", "sales_m_sar")
issues += checks.check_parts_sum_to_total(sectors, national, "sectors", "starting_date", "sales_m_sar")
""")

md("""
## Check 4 — City names across the two sources

The CSV and the PDFs spell some cities differently. The check lists every CSV city with no exact match in the PDFs and the closest names.
A person must confirm each pair is the **same place with the same definition** — for example `Other` (CSV: everything outside 10 cities) and `Others` (PDF: everything outside ~60 cities) look alike but mean different things.
""")
code("""
name_issues = checks.check_name_consistency(
    cities.query("source == 'kapsarc_csv'").city.unique(),
    cities.query("source == 'sama_pdf'").city.unique(),
    "CSV", "PDF", "cities")
issues += name_issues
pd.DataFrame(name_issues)[["key", "evidence", "suggestion"]]
""")

md("""
## Check 5 — Unusual weeks (outliers)

Each week is compared with the median of its neighbouring weeks. Weeks far from it are flagged, then tagged with known events (Ramadan, Eid, Hajj, COVID curfew).
Flagged weeks **with** a known event are listed as `low` (probably real); weeks **without** one are `medium`/`high` and need a person to check the source.
""")
code("""
issues += checks.check_outliers(cities, "cities", "city", "sales_m_sar")
issues += checks.check_outliers(sectors, "sectors", "sectors", "sales_m_sar")
out = pd.DataFrame([i for i in issues if i["check"] == "outlier"])
out.groupby(["table", "severity"]).size().unstack(fill_value=0)
""")

md("## Check 6 — Published % changes match the values (CSV source)")
code("""
pub_city = (published.query("sectors == 'Total' and city != 'Total'")
            .assign(city=lambda d: d.city.str.title())[["starting_date", "city", "value"]]
            .rename(columns={"value": "published_change"}))
chg_issues, n_compared, n_bad = checks.check_published_change(
    csv_cities, pub_city, "cities", "city", "sales_m_sar", "published_change")
issues += chg_issues
print(f"Compared {n_compared} weeks → {n_bad} mismatches")
""")

md("## Review queue")
code("""
report = checks.to_report(issues)
report.to_csv(REVIEW / "issues.csv", index=False)

summary = report.pivot_table(index="check", columns="severity", values="issue_id", aggfunc="count", fill_value=0)
summary = summary.reindex(columns=[c for c in ["high", "medium", "low"] if c in summary.columns])
summary
""")
code("""
# Issues that need a person first: everything that is not 'low'
report.query("severity != 'low'")[["issue_id", "severity", "check", "key", "detail", "evidence", "suggestion"]]
""")

md("""
## How to review an issue

Look at the data before deciding. `plot_issue` draws the weeks around a flagged week, the median it was compared with, and the flagged week in red.

Questions to ask:
- Is there an event that week the checks don't know about (a holiday, a sale, a policy change)?
- Do other cities or sectors move the same way that week? If yes, it is probably real.
- Is the jump a round multiple (×10, ×1000) or a sudden one-week spike with nothing around it? That points to a typing or unit error.
""")
code("""
pending = report.query("severity != 'low'")
for _, issue in pending.query("check == 'outlier'").head(3).iterrows():
    checks.plot_issue(issue, cities, sectors)
""")
code("""
# Review any issue by its id, e.g. show("DQ-0a1614d8")
def show(issue_id):
    checks.plot_issue(report.loc[report.issue_id == issue_id].iloc[0], cities, sectors)

# Example — every city that jumped in the week of 27 Sep 2026, side by side:
week = "2026-09-27"
same_week = report[report.key.str.endswith(week)]
same_week[["issue_id", "key", "detail"]]
""")

md("""
## Decisions template

One row per issue. The reviewer fills `decision` using these words:

| Check | Allowed decisions |
|---|---|
| `name_mismatch` | `map` (use the name in `value`) or `separate` (different places) |
| `outlier` | `keep` (real) or `exclude` (flag the week so analysis skips it — nothing is deleted) |
| `missing_weeks` | `document` (no data to add) |

To save time, `low` outliers are pre-filled `keep` and missing weeks `document`; name issues get the closest name pre-filled in `value` but the decision is left to the reviewer.
""")
code("""
dec_path = REVIEW / "decisions.csv"
def prefill(r):
    if r["check"] == "outlier" and r["severity"] == "low":
        return "keep"
    if r["check"] == "missing_weeks":
        return "document"
    return ""

template = report[["issue_id", "check", "key", "severity", "detail", "suggestion"]].copy()
template["decision"] = template.apply(prefill, axis=1)
template["value"] = [ev.split("'")[1] if c == "name_mismatch" and "'" in ev else ""
                     for c, ev in zip(report["check"], report["evidence"])]
template["note"], template["reviewer"] = "", ""
if dec_path.exists():  # keep decisions already made; add new issues only
    old = pd.read_csv(dec_path, dtype=str).fillna("")
    template = pd.concat([old, template[~template.issue_id.isin(old.issue_id)]], ignore_index=True)
template.to_csv(dec_path, index=False)
print(f"{(template.decision == '').sum()} issues waiting for a decision → {dec_path}")
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "02_data_quality_review.ipynb")
