#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry (Python 3.13, stdlib only, deterministic).

Forecast the top 5 new pump.fun coins by market cap at the answer reads.
Reads $MAKEFUN_INPUT_DIR (else /input/) history.ndjson + task.json,
writes forecast.json to $MAKEFUN_OUTPUT_DIR (else /output/).

Model (stage-2 design):
- Per coin, OLS slope of log10(cap in SOL) vs mark index, shrunk toward
  the population slope by n/(n+4).
- Projected mean = last log cap + shrunk slope * min(horizon_steps, 6)
  + age decay + venue/mayhem adjustment + 0.45 * (3.10 - last log cap).
- Future log cap ~ Normal(projected mean, sigma), sigma from residual
  variance + 0.35^2 + (0.05*sqrt(horizon_steps))^2, floored at 0.35.
- Monte Carlo (seed 12345, 4000 draws) over candidates (top 30 by
  projected cap + top 10 by slope) gives place frequencies.
- Unnamed share: 0.55 at place 1, 0.50 at places 2-5.
- Size bins from the simulated k-th order statistic, normal CDF over
  task.json edges (sd floored at 0.25, capped at 0.50), scaled by 0.95.
- top5 by blended score 0.6 * projected mean + 0.4 * last log cap.
Deterministic: fixed seed, sorted tie-breaks, no time / hash order.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

SEED = 12345
N_DRAWS = 4000
REVERSION_TARGET = 3.10
REVERSION_W = 0.45
SLOPE_CAP_STEPS = 6
BASE_SD = 0.35
NAMED_SHARE = {1: 0.45, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50}
TOP_PER_PLACE = 10
MAX_P = 0.35
SIZE_SCALE = 0.95
TOP5_BLEND_PROJECTED = 0.6


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def log_cap_sol(row):
    return math.log10(int(row["market_cap_lamports"]) / 1e9)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def floor6(p):
    return math.floor(p * 1_000_000) / 1_000_000


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    output_name = task["output"]
    reads = [parse_time(r) for r in task["reads"]]

    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                marks.append(json.loads(line))
    marks = [m for m in marks if m.get("ok") is True and isinstance(m.get("rows"), list)]
    if not marks:
        raise ValueError("the history holds no ok mark")
    last_mark = marks[-1]
    last_time = parse_time(last_mark["at"])
    # Horizon: median answer read minus last mark, in 10-minute steps.
    sorted_reads = sorted(reads)
    median_read = sorted_reads[len(sorted_reads) // 2]
    horizon_seconds = (median_read - last_time).total_seconds()
    horizon_steps = max(0.0, horizon_seconds / 600.0)
    eff_steps = min(horizon_steps, float(SLOPE_CAP_STEPS))

    # Collect per-coin observations: (mark_index, log cap, venue, complete, mayhem, created_at)
    obs = {}
    for idx, mark in enumerate(marks):
        for row in mark.get("rows") or []:
            mint = row["mint"]
            obs.setdefault(mint, []).append(
                (
                    idx,
                    log_cap_sol(row),
                    row.get("venue", "curve"),
                    bool(row.get("complete", False)),
                    bool(row.get("mayhem", False)),
                    row.get("created_at"),
                )
            )

    # Population slope: mean OLS slope over coins with >= 3 observations.
    def ols_slope(xs, ys):
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    slopes = []
    for mint in sorted(obs):
        pts = obs[mint]
        if len(pts) >= 3:
            slopes.append(ols_slope([p[0] for p in pts], [p[1] for p in pts]))
    pop_slope = sum(slopes) / len(slopes) if slopes else 0.0

    # Project each coin.
    projected = {}  # mint -> (mean, sigma, last_log, slope_shrunk, n)
    for mint in sorted(obs):
        pts = sorted(obs[mint], key=lambda p: p[0])
        n = len(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        last_log = ys[-1]
        last_venue = pts[-1][2]
        last_complete = pts[-1][3]
        last_mayhem = pts[-1][4]
        created_at = pts[-1][5]
        if n >= 2:
            slope = ols_slope(xs, ys)
        else:
            slope = pop_slope
        slope_shrunk = (n / (n + 4.0)) * slope + (4.0 / (n + 4.0)) * pop_slope
        # Residual sd around the OLS line.
        if n >= 3:
            mx = sum(xs) / n
            my = sum(ys) / n
            resid_var = sum(
                (y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys)
            ) / n
            resid_sd = math.sqrt(resid_var)
        else:
            resid_sd = 0.08
        # Age decay: -0.015 per hour over 1h old, capped at -0.15.
        age_decay = 0.0
        if created_at:
            try:
                age_h = (last_time - parse_time(created_at)).total_seconds() / 3600.0
                if age_h > 1.0:
                    age_decay = -0.015 * min(age_h - 1.0, 10.0)
            except ValueError:
                age_decay = 0.0
        venue_adj = 0.0
        if last_venue == "curve":
            venue_adj = -0.12
        elif last_venue == "migrating":
            venue_adj = -0.05
        if last_mayhem:
            venue_adj -= 0.03
        if last_complete:
            venue_adj += 0.02
        mean = (
            last_log
            + slope_shrunk * eff_steps
            + age_decay
            + venue_adj
            + REVERSION_W * (REVERSION_TARGET - last_log)
        )
        sigma = math.sqrt(
            resid_sd * resid_sd
            + BASE_SD * BASE_SD
            + (0.05 * math.sqrt(horizon_steps)) ** 2
        )
        sigma = min(max(sigma, 0.35), 0.80)
        projected[mint] = (mean, sigma, last_log, slope_shrunk, n)

    # Candidates: top 30 by projected mean + top 10 by shrunk slope.
    by_mean = sorted(
        projected.items(), key=lambda kv: (-kv[1][0], kv[0])
    )
    by_slope = sorted(
        projected.items(), key=lambda kv: (-kv[1][3], kv[0])
    )
    cand_mints = []
    seen = set()
    for mint, _ in by_mean[:30]:
        if mint not in seen:
            cand_mints.append(mint)
            seen.add(mint)
    for mint, _ in by_slope[:10]:
        if mint not in seen:
            cand_mints.append(mint)
            seen.add(mint)
    # Deterministic order for simulation: projected mean desc, mint asc.
    cand_mints.sort(key=lambda m: (-projected[m][0], m))

    # Monte Carlo.
    rng = random.Random(SEED)
    place_counts = {k: {} for k in range(1, places + 1)}
    order_stats = {k: [] for k in range(1, places + 1)}
    for _ in range(N_DRAWS):
        draws = []
        for mint in cand_mints:
            mean, sigma, _, _, _ = projected[mint]
            draws.append((rng.gauss(mean, sigma), mint))
        draws.sort(key=lambda t: (-t[0], t[1]))
        for k in range(1, places + 1):
            if k - 1 < len(draws):
                logv, mint = draws[k - 1]
                place_counts[k][mint] = place_counts[k].get(mint, 0) + 1
                order_stats[k].append(logv)

    # Place probabilities, scaled to the named share per place.
    probabilities = []
    for k in range(1, places + 1):
        target = NAMED_SHARE[k]
        counts = place_counts[k]
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_PER_PLACE]
        if not ranked:
            continue
        # Shrink frequencies 0.9 toward uniform over the kept coins, then scale to target.
        n_kept = len(ranked)
        raw = []
        for mint, c in ranked:
            freq = c / N_DRAWS
            raw.append((mint, 0.9 * freq + 0.1 * target / n_kept))
        total_raw = sum(p for _, p in raw)
        scaled = [(m, p * target / total_raw) for m, p in raw]
        # Cap at MAX_P once, then rescale the rest to fill the target.
        capped = [(m, min(p, MAX_P)) for m, p in scaled]
        sum_capped = sum(p for _, p in capped)
        if sum_capped < target - 1e-12:
            room = [(m, p) for m, p in capped if p < MAX_P - 1e-12]
            room_sum = sum(p for _, p in room)
            if room_sum > 0:
                need = target - sum_capped
                capped = [
                    (m, p + need * p / room_sum) if p < MAX_P - 1e-12 else (m, p)
                    for m, p in capped
                ]
        for mint, p in capped:
            fp = floor6(p)
            if fp > 0:
                probabilities.append({"mint": mint, "place": k, "p": fp})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))

    # Top5 by blended score 0.6 * projected mean + 0.4 * last log cap,
    # so slope-chasers with a low last cap cannot jump to #1 on slope alone.
    by_blended = sorted(
        projected.items(),
        key=lambda kv: (
            -(TOP5_BLEND_PROJECTED * kv[1][0] + (1.0 - TOP5_BLEND_PROJECTED) * kv[1][2]),
            kv[0],
        ),
    )
    top5 = [mint for mint, _ in by_blended[:places]]

    # Size bins from simulated order statistics.
    sizes = []
    for k in range(1, places + 1):
        vals = order_stats[k]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        sd = min(max(math.sqrt(var), 0.25), 0.50)
        for b in range(0, len(edges) + 1):
            if b == 0:
                lo = float("-inf")
                hi = edges[0]
            elif b == len(edges):
                lo = edges[-1]
                hi = float("inf")
            else:
                lo = edges[b - 1]
                hi = edges[b]
            cdf_lo = 0.0 if lo == float("-inf") else norm_cdf((lo - mean) / sd)
            cdf_hi = 1.0 if hi == float("inf") else norm_cdf((hi - mean) / sd)
            p = SIZE_SCALE * (cdf_hi - cdf_lo)
            fp = floor6(p)
            if fp > 0:
                sizes.append({"place": k, "bin": b, "p": fp})
    sizes.sort(key=lambda r: (r["place"], r["bin"]))

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
