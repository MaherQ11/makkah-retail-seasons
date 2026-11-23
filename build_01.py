import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Makkah Retail Sales — 01: Structural Cleaning

**Question:** How do Ramadan and Hajj change card spending in Makkah, and how much should we expect in the next season?

This notebook builds one clean weekly dataset from two public sources of SAMA (Saudi Central Bank) point-of-sale data:

| Source | Format | Coverage |
|---|---|---|
| [KAPSARC Data Portal](https://datasource.kapsarc.org/explore/dataset/point-of-sale-transactions-by-sector-and-city/) | CSV | May 2020 – Jul 2025, 11 cities, 17 sectors |
| [SAMA weekly reports](https://www.sama.gov.sa/en-US/Statistics/Indices/Pages/POS.aspx) | PDF (Arabic/English) | Selected weeks Aug 2025 – Nov 2026 |

This notebook fixes the **structure** of the data (layout, dates, units, merging sources). Fixes that need judgement — unusual values, names that may or may not be the same place — are **not** made here: notebook 02 finds them and a person decides.

Pipeline: `01 structural cleaning` → `02 quality review (automatic checks)` → **human decisions** → `03 apply decisions`.
""")

code("""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
from hijridate import Gregorian

RAW = Path("data/raw")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

LOG = []  # data-quality log: one line per issue found and how it was handled
def log(issue, action, rows=None):
    LOG.append({"issue": issue, "action": action, "rows_affected": rows})
    print(f"• {issue}\\n  → {action}" + (f" ({rows} rows)" if rows is not None else ""))
""")

md("## 1. Load and profile the KAPSARC file")
code("""
raw = pd.read_csv(RAW / "kapsarc_pos.csv", parse_dates=["starting_date"])
print(raw.shape)
print("Missing values:", int(raw.isna().sum().sum()))
print("Duplicate keys:", int(raw.duplicated(["starting_date", "number_value_change_transactions", "sectors", "city"]).sum()))
raw.head()
""")
code("""
for col in ["number_value_change_transactions", "city", "sectors"]:
    print(col, "→", raw[col].nunique(), "values")
raw["number_value_change_transactions"].value_counts()
""")

md("""
## 2. Issue: four different measures share one column

The `value` column holds **sales value**, **number of transactions**, and two **week-over-week % changes**.
Mixing them makes any sum meaningless. We keep the two base measures as separate columns; the % changes can be recomputed from them, so they are dropped.
""")
code("""
MEASURES = {
    "Value of Transactions (In Thousand SAR)": "value_k_sar",
    "Number of Transactions (In Thousand)": "count_k",
}
dropped = (~raw["number_value_change_transactions"].isin(MEASURES)).sum()
df = (raw[raw["number_value_change_transactions"].isin(MEASURES)]
      .assign(measure=lambda d: d["number_value_change_transactions"].map(MEASURES))
      .pivot_table(index=["starting_date", "city", "sectors"], columns="measure", values="value")
      .reset_index())
df.columns.name = None
log("Value, count and two % change measures stored in one column",
    "Kept value and count as separate columns; dropped % changes (recomputable)", int(dropped))
df.head()
""")

md("""
## 3. Issue: totals are mixed in with the detail rows

Each row is either a **city** (with sector = `Total`) or a **sector** (with city = `Total`). There is no city × sector breakdown, and a `Total × Total` row holds the national figure.
Summing everything would count every riyal three times. We split the data into three tables and **check that the parts add up to the total**.
""")
code("""
national = df[(df.city == "Total") & (df.sectors == "Total")].drop(columns=["city", "sectors"]).set_index("starting_date")
by_city = df[(df.city != "Total") & (df.sectors == "Total")].drop(columns="sectors")
by_sector = df[(df.city == "Total") & (df.sectors != "Total")].drop(columns="city")

for name, part, key in [("cities", by_city, "city"), ("sectors", by_sector, "sectors")]:
    s = part.groupby("starting_date")["value_k_sar"].sum()
    gap = ((s - national["value_k_sar"]) / national["value_k_sar"]).abs().max() * 100
    print(f"Sum of {name} vs national total: max difference {gap:.5f}%")

log("City rows, sector rows and national totals mixed in one table",
    "Split into national / by-city / by-sector tables; parts add up to the total (difference < 0.001%)")
""")

md("""
## 4. Issue: one week starts on the wrong day

SAMA weeks run Sunday to Saturday. One week is dated **Saturday 20 June 2020**, which leaves a 6-day gap before it and an 8-day gap after it.
It is the week starting Sunday 21 June, entered one day early.
""")
code("""
dates = pd.Series(sorted(df.starting_date.unique()))
odd = dates[dates.dt.dayofweek != 6]  # 6 = Sunday
print(odd.dt.strftime("%Y-%m-%d (%A)").tolist())

fix = {pd.Timestamp("2020-06-20"): pd.Timestamp("2020-06-21")}
n = 0
for t in (by_city, by_sector):
    n += int(t.starting_date.isin(fix).sum())
    t["starting_date"] = t["starting_date"].replace(fix)
national.index = national.index.to_series().replace(fix)
assert (pd.Series(sorted(by_city.starting_date.unique())).diff().dropna().dt.days == 7).all()
log("Week dated Saturday 2020-06-20 (all other weeks start on Sunday)", "Moved to Sunday 2020-06-21; all gaps are now exactly 7 days", int(n))
""")

md("""
## 5. Issue: values are in thousands

`530,718` means **530.7 million SAR**. We convert to SAR millions and millions of transactions, which read naturally in charts.
""")
code("""
def to_millions(t):
    t = t.copy()
    t["sales_m_sar"] = t.pop("value_k_sar") / 1_000
    t["transactions_m"] = t.pop("count_k") / 1_000
    return t

national, by_city, by_sector = map(to_millions, (national, by_city, by_sector))
by_city["city"] = by_city["city"].str.title()  # formatting only; matching names across sources is a review decision (notebook 02)
log("Values in thousands of SAR / thousands of transactions", "Converted to SAR millions / millions of transactions")
by_city.query("city == 'Makkah'").head()
""")

md("""
## 6. Issue: lockdown weeks are not normal trading

The first weeks of the data (May–June 2020) fall under COVID-19 curfews, and the week of 24 May 2020 (Eid al-Fitr under a full curfew) is the lowest on record.
We **keep** these weeks but **flag** them, so the analysis and forecast can exclude them.
""")
code("""
COVID_END = pd.Timestamp("2020-06-21")  # nationwide curfew lifted
by_city["covid_period"] = by_city.starting_date < COVID_END
log("Weeks before 2020-06-21 affected by COVID-19 curfews", "Flagged with covid_period=True (kept, excluded later)",
    int(by_city.query("city == 'Makkah'").covid_period.sum()))
""")

md("""
## 7. Extract the newer weeks from SAMA's PDF reports

After July 2025 the data is only published as weekly PDF reports. Each report covers **4 weeks** and lists ~60 cities, with English and Arabic names on the same line.
We read the city tables, keep the English name and the 8 numbers (count and value for 4 weeks), and ignore the % change columns.
""")
code("""
DATE_RE = re.compile(r"(\\d{1,2} \\w{3},\\d{2}) - (\\d{1,2} \\w{3},\\d{2})")
NUM_RE = re.compile(r"-?[\\d,]+(?:\\.\\d+)?")

def parse_report(path):
    rows = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if "By Cities" not in text:
                continue
            weeks = [pd.to_datetime(a, format="%d %b,%y") for a, _ in DATE_RE.findall(text)[:4]]
            for line in text.splitlines():
                name = re.match(r"^([A-Z][A-Za-z'\\-]+)", line)
                nums = NUM_RE.findall(line)
                if not name or len(nums) < 10:
                    continue
                vals = [float(n.replace(",", "")) for n in nums[-10:-2]]  # last 2 = % changes
                for i, wk in enumerate(weeks):
                    rows.append({"starting_date": wk, "city": name.group(1),
                                 "count_k": vals[2 * i], "value_k_sar": vals[2 * i + 1], "source": path.name})
    return pd.DataFrame(rows)

reports = sorted((RAW / "sama_pdfs").glob("*.pdf"))
sama = pd.concat([parse_report(p) for p in reports], ignore_index=True)
print(f"{len(reports)} reports → {len(sama)} rows, {sama.city.nunique()} cities, "
      f"{sama.starting_date.nunique()} weeks")
""")

md("""
### Issue: the same week appears in two reports

Reports overlap (e.g. the week of 23 Aug 2026 is in both the 29 Aug and 19 Sep reports). We check that the overlapping figures agree, then keep one copy.
""")
code("""
dup = sama.duplicated(["starting_date", "city"], keep=False)
check = sama[dup].groupby(["starting_date", "city"])["value_k_sar"].nunique()
print("Overlapping weeks:", sorted(sama[dup].starting_date.dt.strftime("%Y-%m-%d").unique()))
print("Do the duplicates agree?", bool((check == 1).all()))
n_dup = int(sama.duplicated(["starting_date", "city"]).sum())
sama = sama.drop_duplicates(["starting_date", "city"]).drop(columns="source")
log("Same week published in two overlapping PDF reports", "Checked the figures match, kept one copy", n_dup)

sama = to_millions(sama)
sama["covid_period"] = False
sama.query("city == 'Makkah'").sort_values("starting_date")
""")

md("""
### Issue: SAMA changed its city breakdown

The CSV has 11 cities plus an `Other` group; the newer PDFs list ~60 cities. **Makkah** is reported on its own in both, so its series can be joined. National-level comparisons across the two formats are not attempted.
""")
code("""
log("City breakdown changed from 11 cities + 'Other' (CSV) to ~60 cities (PDF)",
    "Combined both, but use only Makkah across the two formats; checked its level is consistent at the join")
cities = pd.concat([by_city.assign(source="kapsarc_csv"), sama.assign(source="sama_pdf")], ignore_index=True)
makkah = cities.query("city == 'Makkah'").sort_values("starting_date").reset_index(drop=True)

# Consistency at the join: last CSV weeks vs first PDF weeks (same season, outside Ramadan/Hajj)
join = pd.concat([makkah.query("source == 'kapsarc_csv'").tail(4), makkah.query("source == 'sama_pdf'").head(4)])
print(join[["starting_date", "source", "sales_m_sar", "transactions_m"]])
""")

md("""
## 8. Coverage: which weeks are missing?

SAMA's website only lists its 10 latest reports, so some periods could not be collected. We make the gaps explicit instead of hiding them.
""")
code("""
full = pd.date_range(makkah.starting_date.min(), makkah.starting_date.max(), freq="W-SUN")
have = set(makkah.starting_date)
missing = pd.Series([d for d in full if d not in have])
gaps = (missing.groupby((missing.diff() != pd.Timedelta(days=7)).cumsum())
        .agg(["min", "max", "count"]).rename(columns={"min": "from", "max": "to", "count": "weeks"}))
print(f"{len(have)} of {len(full)} weeks available ({len(have) / len(full):.0%})")
log("Weeks missing between Jul 2025 and Nov 2026 (reports not archived online)",
    "Listed in coverage table; not filled in", int(len(missing)))
gaps
""")

md("## 9. Save the interim data and the log")
code("""
cities.to_csv(OUT / "cities_weekly.csv", index=False)
by_sector.to_csv(OUT / "sectors_weekly.csv", index=False)
national.reset_index().to_csv(OUT / "national_weekly.csv", index=False)
gaps.to_csv(OUT / "coverage_gaps.csv", index=False)

# % changes published by the source, kept only to cross-check our values in notebook 02
chg = raw[raw.number_value_change_transactions == "Value of Transactions Change %"]
chg = chg.assign(starting_date=chg.starting_date.replace(fix))
chg.to_csv(OUT / "published_changes.csv", index=False)

log_df = pd.DataFrame(LOG)
log_df.to_csv(OUT / "cleaning_log.csv", index=False)
log_df
""")

md("""
## Next

`02_data_quality_review.ipynb` runs automatic checks on this interim data and lists every issue for a person to review.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "01_structural_cleaning.ipynb")
