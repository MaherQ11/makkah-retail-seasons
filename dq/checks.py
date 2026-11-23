"""Data-quality checks for weekly point-of-sale data.

Every check is a function that takes a DataFrame and returns a list of issues.
Checks only *find and describe* problems; they never change the data.
A person reviews the issues and records a decision for each one in decisions.csv.

Issue fields
------------
check       name of the check that raised it
table       which table the issue is in
key         what the issue is about (a date, a city, a sector, a pair of names ...)
detail      what was found, in plain words
evidence    the numbers behind it
severity    high   = likely an error, blocks use of the data
            medium = needs a human look
            low    = probably real (e.g. explained by Ramadan), listed for completeness
suggestion  what the check suggests doing
"""
from __future__ import annotations

import difflib
import hashlib
import numpy as np
import pandas as pd
from hijridate import Gregorian

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _issue(check, table, key, detail, evidence, severity, suggestion):
    return dict(check=check, table=table, key=str(key), detail=detail, evidence=evidence,
                severity=severity, suggestion=suggestion)


# ---------------------------------------------------------------- calendar ---
def calendar_flags(dates) -> pd.DataFrame:
    """Hijri (Umm al-Qura) flags for each week starting on the given dates."""
    rows = []
    for start in pd.to_datetime(pd.Series(dates).drop_duplicates()):
        days = [Gregorian.fromdate((start + pd.Timedelta(days=i)).date()).to_hijri() for i in range(7)]
        rows.append({
            "starting_date": start,
            "hijri_year": days[0].year,
            "ramadan_days": sum(d.month == 9 for d in days),
            "last_10_ramadan": any(d.month == 9 and d.day >= 20 for d in days),
            "eid_fitr": any(d.month == 10 and d.day <= 3 for d in days),
            "hajj": any(d.month == 12 and 8 <= d.day <= 13 for d in days),
            "eid_adha": any(d.month == 12 and 10 <= d.day <= 12 for d in days),
        })
    return pd.DataFrame(rows)


# Known events on the Gregorian calendar. Add to this list as new explanations are confirmed by a reviewer.
GREGORIAN_EVENTS = [
    # (name, month, first_day, last_day, from_year)
    ("school start", 8, 15, 31, 2020),
    ("Saudi National Day", 9, 20, 26, 2020),
    ("Founding Day", 2, 20, 25, 2022),
    ("White Friday sales", 11, 20, 30, 2020),
    ("year-end holidays", 12, 24, 31, 2020),
]
DATE_RANGES = [
    # (name, first_week_start, last_week_start)
    ("reopening after COVID curfew", "2020-06-21", "2020-07-12"),
]


SALARY_DAY = 27  # Saudi government salaries are paid on the 27th (added after review, Nov 2026)


def _gregorian_events(start) -> list[str]:
    days = [start + pd.Timedelta(days=i) for i in range(7)]
    tags = ["salary week"] if any(d.day == SALARY_DAY for d in days) else []
    tags += [name for name, m, d1, d2, y in GREGORIAN_EVENTS
            if any(d.month == m and d1 <= d.day <= d2 and d.year >= y for d in days)]
    tags += [name for name, a, b in DATE_RANGES if pd.Timestamp(a) <= start <= pd.Timestamp(b)]
    return tags


def _context(row) -> str:
    """Known events that can explain an unusual week."""
    tags = _gregorian_events(row["starting_date"])
    if row.get("covid_period"):
        tags.append("COVID curfew")
    if row.get("last_10_ramadan"):
        tags.append("last 10 days of Ramadan")
    elif row.get("ramadan_days", 0) > 0:
        tags.append("Ramadan")
    if row.get("eid_fitr"):
        tags.append("Eid al-Fitr")
    if row.get("hajj"):
        tags.append("Hajj")
    if row.get("eid_adha"):
        tags.append("Eid al-Adha")
    return ", ".join(tags)


# ------------------------------------------------------------------ checks ---
def check_missing_and_duplicates(df, table, keys):
    issues = []
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        cols = df.columns[df.isna().any()].tolist()
        issues.append(_issue("missing_values", table, ",".join(cols), "Empty cells found",
                             f"{n_missing} empty cells", "high", "Find out why they are empty before filling"))
    dup = df[df.duplicated(keys, keep=False)]
    if len(dup):
        issues.append(_issue("duplicates", table, ",".join(keys), "Same key appears more than once",
                             f"{len(dup)} rows", "high", "Check if copies agree; keep one"))
    return issues


def check_week_regularity(df, table, date_col="starting_date", weekday=6):
    """Weeks should start on the same weekday and be 7 days apart."""
    issues = []
    dates = pd.Series(sorted(pd.to_datetime(df[date_col]).unique()))
    for d in dates[dates.dt.dayofweek != weekday]:
        issues.append(_issue("week_start_day", table, d.date(), f"Week starts on {d.day_name()}, not Sunday",
                             "", "high", "Move to the nearest Sunday if the 7-day spacing confirms it"))
    gaps = dates.diff().dt.days
    for d, g in zip(dates[gaps > 7], gaps[gaps > 7]):
        issues.append(_issue("missing_weeks", table, d.date(), f"{int(g // 7) - 1} week(s) missing before this date",
                             f"gap of {int(g)} days", "medium", "Look for the missing reports; otherwise document the gap"))
    return issues


def check_parts_sum_to_total(parts, total, table, by, value_col, tolerance_pct=0.01):
    s = parts.groupby(by)[value_col].sum()
    diff = ((s - total[value_col]) / total[value_col] * 100).abs().dropna()
    bad = diff[diff > tolerance_pct]
    return [_issue("parts_vs_total", table, k, "Parts do not add up to the total", f"{v:.3f}% difference",
                   "high", "Check for missing or double-counted rows") for k, v in bad.items()]


def _name_key(name: str) -> str:
    """Spelling-insensitive key: lower case, no 'Al-' prefix, letters only."""
    n = name.lower()
    n = n[3:] if n.startswith("al-") else n
    return "".join(ch for ch in n if ch.isalpha())


def check_name_consistency(names_a, names_b, label_a, label_b, table, cutoff=0.6):
    """Names in one source with no exact match in the other, plus the closest candidates."""
    issues = []
    only_a = sorted(set(names_a) - set(names_b))
    pool = {_name_key(n): n for n in set(names_b) - set(names_a)}
    for name in only_a:
        close = [pool[k] for k in difflib.get_close_matches(_name_key(name), list(pool), n=3, cutoff=cutoff)]
        if close:
            issues.append(_issue("name_mismatch", table, name, f"'{name}' ({label_a}) has no exact match in {label_b}",
                                 f"closest: {close}", "medium",
                                 f"Map '{name}' → '{close[0]}' if they are the same place; confirm the definitions match"))
        else:
            issues.append(_issue("name_mismatch", table, name, f"'{name}' ({label_a}) has no match in {label_b}",
                                 "no close name", "low", "Keep separate"))
    return issues


def check_outliers(df, table, entity_col, value_col, window=9, z_threshold=3.5, min_points=20):
    """Weeks far from their neighbours (robust z-score on log values vs a rolling median).

    Each flagged week is tagged with known events. Weeks explained by an event get severity 'low';
    the rest are 'medium' (or 'high' when very extreme) and need a human look.
    """
    issues = []
    for entity, g in df.sort_values("starting_date").groupby(entity_col):
        g = g.dropna(subset=[value_col])
        if len(g) < min_points:
            continue
        y = np.log(g[value_col].clip(lower=1e-9))
        resid = y - y.rolling(window, center=True, min_periods=3).median()
        mad = 1.4826 * (resid - resid.median()).abs().median()
        if not mad:
            continue
        z = (resid - resid.median()) / mad
        for idx in z[z.abs() > z_threshold].index:
            row = g.loc[idx]
            ctx = _context(row)
            pct = (np.exp(resid[idx]) - 1) * 100
            severity = "low" if ctx else ("high" if abs(z[idx]) > 8 else "medium")
            issues.append(_issue(
                "outlier", table, f"{entity} | {row['starting_date']:%Y-%m-%d}",
                f"{'Above' if pct > 0 else 'Below'} its neighbouring weeks by {abs(pct):.0f}%"
                + (f" — during {ctx}" if ctx else " — no known event this week"),
                f"value={row[value_col]:,.1f}, z={z[idx]:.1f}", severity,
                "Probably real; keep" if ctx else "Check the source report for this week"))
    return issues


def check_published_change(base, published_change, table, entity_col, value_col, change_col, tolerance=0.15):
    """Recompute week-over-week % change and compare with the % change the source published."""
    issues = []
    b = base.sort_values("starting_date").copy()
    b["recomputed"] = b.groupby(entity_col)[value_col].pct_change() * 100
    m = b.merge(published_change, on=["starting_date", entity_col], how="inner")
    m["gap"] = (m["recomputed"] - m[change_col]).abs()
    bad = m[m["gap"] > tolerance]
    for _, r in bad.iterrows():
        issues.append(_issue("published_change_mismatch", table, f"{r[entity_col]} | {r['starting_date']:%Y-%m-%d}",
                             "Published % change does not match the values",
                             f"published {r[change_col]:.1f}% vs recomputed {r['recomputed']:.1f}%",
                             "medium", "One of the two numbers is wrong in the source; check the report"))
    return issues, len(m), len(bad)


# ------------------------------------------------------------------ report ---
def to_report(issues) -> pd.DataFrame:
    rep = pd.DataFrame(issues, columns=["check", "table", "key", "detail", "evidence", "severity", "suggestion"])
    if rep.empty:
        return rep
    rep = rep.sort_values(["severity", "check", "key"], key=lambda s: s.map(SEVERITY_ORDER) if s.name == "severity" else s)
    # Stable id: the same issue keeps the same id when the checks are run again
    rep.insert(0, "issue_id", [
        "DQ-" + hashlib.md5(f"{c}|{t}|{k}".encode()).hexdigest()[:8]
        for c, t, k in zip(rep["check"], rep["table"], rep["key"])])
    return rep.reset_index(drop=True)


# ------------------------------------------------------------- review aid ---
def plot_issue(issue: pd.Series, cities: pd.DataFrame, sectors: pd.DataFrame, weeks_around: int = 12):
    """Show the data behind an outlier issue so a reviewer can judge it.

    Plots the series ±`weeks_around` weeks around the flagged week, marks the week in red,
    and draws the rolling median the check compared it with.
    """
    import matplotlib.pyplot as plt

    if issue["check"] != "outlier":
        print(f"{issue['issue_id']}: {issue['detail']}  |  {issue['evidence']}")
        return
    entity, week = issue["key"].split(" | ")
    week = pd.Timestamp(week)
    table = cities if entity in set(cities.city) else sectors
    col = "city" if table is cities else "sectors"
    s = table[table[col] == entity].set_index("starting_date").sales_m_sar.sort_index()
    window = s[week - pd.Timedelta(weeks=weeks_around): week + pd.Timedelta(weeks=weeks_around)]
    med = s.rolling(9, center=True, min_periods=3).median().reindex(window.index)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(window.index, window.values, marker="o", ms=3, label="weekly sales")
    ax.plot(med.index, med.values, ls="--", color="gray", label="neighbouring-weeks median")
    ax.scatter([week], [s.get(week)], color="red", zorder=5, s=60, label="flagged week")
    ax.set_title(f"{issue['issue_id']} · {entity} · {week:%Y-%m-%d}\n{issue['detail']}", loc="left", fontsize=10)
    ax.set_ylabel("SAR millions"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.show()
