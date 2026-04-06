"""
SPY Day-Type Predictor
======================
Analiza un CSV de datos horarios del SPY, clasifica cada día en categorías
interpretables y calcula la probabilidad de que el día siguiente suba o baje.

Genera un reporte HTML autocontenido con gráficos interactivos.

Uso:
    python spy_analysis.py datos.csv
    python spy_analysis.py datos.csv --output reporte.html
    python spy_analysis.py datos.csv --umbral-ret -0.3 --umbral-gap 0.15

Dependencias:
    pip install pandas numpy scipy scikit-learn
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_build_daily(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    # Normalizar timezone → NY
    df.index = df.index.tz_convert("America/New_York")

    df["date"] = df.index.date

    days = []
    for date, g in df.groupby("date"):
        g = g.sort_index()
        if len(g) < 4:
            continue
        o = g["open"].iloc[0]
        c = g["close"].iloc[-1]
        h = g["high"].max()
        l = g["low"].min()
        mid = len(g) // 2

        days.append({
            "date":          date,
            "open":          o,
            "close":         c,
            "high":          h,
            "low":           l,
            "volume":        g["volume"].sum(),
            "ret":           (c - o) / o * 100,
            "range_pct":     (h - l) / o * 100,
            "first_bar_ret": (g["close"].iloc[0] - o) / o * 100,
            "last_bar_ret":  (c - g["open"].iloc[-1]) / g["open"].iloc[-1] * 100,
            "morning_ret":   (g["close"].iloc[mid - 1] - o) / o * 100,
            "afternoon_ret": (c - g["close"].iloc[mid - 1]) / g["close"].iloc[mid - 1] * 100,
            "close_pos":     (c - l) / (h - l) if h != l else 0.5,
            "vol_trend":     (g["volume"].iloc[mid:].sum() - g["volume"].iloc[:mid].sum())
                             / (g["volume"].iloc[:mid].sum() + 1),
        })

    daily = pd.DataFrame(days).set_index("date")
    daily["gap"]     = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1) * 100
    daily["vol_rel"] = daily["volume"] / daily["volume"].rolling(20, min_periods=5).mean()
    daily["next_ret"] = daily["ret"].shift(-1)
    daily["up"]       = (daily["next_ret"] > 0).astype(float)

    features = ["ret", "range_pct", "gap", "close_pos", "first_bar_ret",
                "morning_ret", "afternoon_ret", "last_bar_ret", "vol_rel", "vol_trend"]

    daily_clean = daily.dropna(subset=features + ["next_ret"]).copy()
    return daily, daily_clean, features


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(p, n, confidence=0.90):
    z = norm.ppf((1 + confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(center - margin, 3), round(center + margin, 3)


def analyze_group(daily, mask, name):
    g = daily[mask]
    if len(g) < 8:
        return None
    p  = g["up"].mean()
    n  = len(g)
    ci = wilson_ci(p, n)
    _, pval = ttest_1samp(g["up"].values, 0.5)
    base = daily["up"].mean()
    return {
        "name":           name,
        "n":              n,
        "pct_days":       round(n / len(daily) * 100, 1),
        "p_up":           round(p, 4),
        "ci_lo":          ci[0],
        "ci_hi":          ci[1],
        "pval":           round(pval, 3),
        "edge":           round(abs(p - base), 4),
        "next_ret_mean":  round(g["next_ret"].mean(), 4),
        "next_ret_std":   round(g["next_ret"].std(), 4),
        "ret_mean":       round(g["ret"].mean(), 4),
        "range_mean":     round(g["range_pct"].mean(), 4),
        "gap_mean":       round(g["gap"].mean(), 4),
        "close_pos_mean": round(g["close_pos"].mean(), 4),
    }


def run_analysis(daily_clean, features, cfg):
    N    = len(daily_clean)
    base = daily_clean["up"].mean()

    # ── k-means (para el gráfico de scores) ──────────────────────────────────
    X  = daily_clean[features].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    kmeans_scores = []
    for n in range(2, 12):
        km  = KMeans(n_clusters=n, random_state=42, n_init=20)
        lbl = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lbl)
        daily_clean["_c"] = lbl
        probs  = daily_clean.groupby("_c")["up"].mean()
        sizes  = daily_clean.groupby("_c").size()
        min_n  = sizes.min()
        w_edge = sum(abs(probs[c] - 0.5) * sizes[c] for c in probs.index) / N
        size_ok = 1.0 if min_n >= 10 else (0.7 if min_n >= 6 else 0.3)
        score   = (sil * 0.35 + w_edge * 0.65) * size_ok
        kmeans_scores.append({"n": n, "score": round(score, 4)})

    daily_clean.drop(columns=["_c"], inplace=True, errors="ignore")

    # ── Segmentos de una sola feature ────────────────────────────────────────
    single_cuts = {
        "ret":           [(-99, -1), (-1, -0.3), (-0.3, 0.3), (0.3, 1), (1, 99)],
        "gap":           [(-99, -0.3), (-0.3, 0.3), (0.3, 99)],
        "close_pos":     [(0, 0.25), (0.25, 0.45), (0.45, 0.55), (0.55, 0.75), (0.75, 1.01)],
        "range_pct":     [(0, 0.5), (0.5, 0.9), (0.9, 1.5), (1.5, 99)],
        "morning_ret":   [(-99, -0.3), (-0.3, 0.3), (0.3, 99)],
        "afternoon_ret": [(-99, -0.3), (-0.3, 0.3), (0.3, 99)],
        "vol_rel":       [(0, 0.8), (0.8, 1.2), (1.2, 99)],
    }
    single_results = []
    for feat, bins in single_cuts.items():
        for lo, hi in bins:
            mask = (daily_clean[feat] >= lo) & (daily_clean[feat] < hi)
            r = analyze_group(daily_clean, mask, f"{feat} [{lo},{hi})")
            if r:
                single_results.append({**r, "feature": feat, "lo": lo, "hi": hi})
    single_df = pd.DataFrame(single_results).sort_values("edge", ascending=False)

    # ── Señales combinadas (2 features) ──────────────────────────────────────
    two_results = []
    for cp_lo, cp_hi in [(0, 0.35), (0.35, 0.65), (0.65, 1.01)]:
        cp_m = (daily_clean["close_pos"] >= cp_lo) & (daily_clean["close_pos"] < cp_hi)
        for ret_lo, ret_hi in [(-99, -0.3), (-0.3, 0.3), (0.3, 99)]:
            m = cp_m & (daily_clean["ret"] >= ret_lo) & (daily_clean["ret"] < ret_hi)
            r = analyze_group(daily_clean, m, f"close_pos[{cp_lo},{cp_hi}) & ret[{ret_lo},{ret_hi})")
            if r: two_results.append(r)
        for al, ah in [(-99, -0.2), (-0.2, 0.2), (0.2, 99)]:
            m = cp_m & (daily_clean["afternoon_ret"] >= al) & (daily_clean["afternoon_ret"] < ah)
            r = analyze_group(daily_clean, m, f"close_pos[{cp_lo},{cp_hi}) & tarde[{al},{ah})")
            if r: two_results.append(r)

    # señales ret+gap
    for rl, rh in [(-99, -0.3), (-0.3, 0.3), (0.3, 99)]:
        for gl, gh in [(-99, -0.15), (-0.15, 0.15), (0.15, 99)]:
            m = (daily_clean["ret"] >= rl) & (daily_clean["ret"] < rh) & \
                (daily_clean["gap"] >= gl) & (daily_clean["gap"] < gh)
            r = analyze_group(daily_clean, m, f"ret[{rl},{rh}) & gap[{gl},{gh})")
            if r: two_results.append(r)

    two_df = pd.DataFrame(two_results).sort_values("edge", ascending=False)

    # ── Categorías interpretables ─────────────────────────────────────────────
    cats_def = {
        "A: Alcista fuerte":       (daily_clean["close_pos"] >= 0.70) & (daily_clean["ret"] > 0.3),
        "B: Alcista con tarde fort":(daily_clean["close_pos"] >= 0.65) & (daily_clean["afternoon_ret"] > 0.2),
        "C: Lateral":              (daily_clean["close_pos"] >= 0.35) & (daily_clean["close_pos"] < 0.65),
        "D: Bajista tarde débil":  (daily_clean["close_pos"] < 0.35)  & (daily_clean["afternoon_ret"] < -0.2),
        "E: Capitulación":         (daily_clean["close_pos"] < 0.30)  & (daily_clean["ret"] < -0.3),
    }

    daily_clean["cat"] = "F: Resto"
    for name, mask in reversed(list(cats_def.items())):
        daily_clean.loc[mask, "cat"] = name

    cat_summary = []
    for name in list(cats_def.keys()) + ["F: Resto"]:
        r = analyze_group(daily_clean, daily_clean["cat"] == name, name)
        if r:
            cat_summary.append(r)

    # ── Señal combinada especial ──────────────────────────────────────────────
    ret_thr = cfg.get("umbral_ret", -0.3)
    gap_thr = cfg.get("umbral_gap",  0.15)
    combo_mask = (daily_clean["ret"] < ret_thr) & (daily_clean["gap"] > gap_thr)
    combo = analyze_group(daily_clean, combo_mask,
                          f"Retorno<{ret_thr}% y Gap>{gap_thr}%")

    return {
        "N":             N,
        "base_rate":     round(float(base), 4),
        "date_range":    f"{daily_clean.index[0]} → {daily_clean.index[-1]}",
        "kmeans_scores": kmeans_scores,
        "cat_summary":   cat_summary,
        "top_single":    single_df.head(12).to_dict(orient="records"),
        "top_two":       two_df.head(12).to_dict(orient="records"),
        "combo":         combo,
        "ret_thr":       ret_thr,
        "gap_thr":       gap_thr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASIFICAR EL DÍA MÁS RECIENTE
# ─────────────────────────────────────────────────────────────────────────────

def classify_today(daily, daily_clean, cfg):
    today_row = daily.iloc[-1]
    today_date = str(today_row.name)

    cats_def = {
        "A: Alcista fuerte":       today_row["close_pos"] >= 0.70 and today_row["ret"] > 0.3,
        "B: Alcista con tarde fort":today_row["close_pos"] >= 0.65 and today_row.get("afternoon_ret", 0) > 0.2,
        "C: Lateral":              0.35 <= today_row["close_pos"] < 0.65,
        "D: Bajista tarde débil":  today_row["close_pos"] < 0.35  and today_row.get("afternoon_ret", 0) < -0.2,
        "E: Capitulación":         today_row["close_pos"] < 0.30  and today_row["ret"] < -0.3,
    }

    today_cat = "F: Resto"
    for name, cond in cats_def.items():
        if cond:
            today_cat = name
            break

    # buscar probabilidad de la categoría
    cat_probs = {r["name"]: r["p_up"] for r in
                 [analyze_group(daily_clean, daily_clean["cat"] == c, c)
                  for c in list(cats_def.keys()) + ["F: Resto"]]
                 if r is not None}

    today_p = cat_probs.get(today_cat, float(daily_clean["up"].mean()))

    ret_thr = cfg.get("umbral_ret", -0.3)
    gap_thr = cfg.get("umbral_gap",  0.15)
    combo_active = (today_row["ret"] < ret_thr) and (not np.isnan(today_row.get("gap", float("nan"))) and today_row["gap"] > gap_thr)

    return {
        "date":          today_date,
        "cat":           today_cat,
        "p_up":          round(today_p, 4),
        "combo_active":  combo_active,
        "features": {
            "ret":           round(float(today_row["ret"]),           3),
            "range_pct":     round(float(today_row["range_pct"]),     3),
            "gap":           round(float(today_row.get("gap", 0)),    3),
            "close_pos":     round(float(today_row["close_pos"]),     3),
            "morning_ret":   round(float(today_row["morning_ret"]),   3),
            "afternoon_ret": round(float(today_row["afternoon_ret"]), 3),
            "vol_rel":       round(float(today_row.get("vol_rel", 1)),3),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GENERACIÓN DEL HTML
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPY Análisis Predictivo — {date_range}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --bg:   #f9fafb; --surface: #ffffff; --surface2: #f3f4f6;
    --border: #e5e7eb; --border2: #d1d5db;
    --text:  #111827; --muted: #6b7280; --hint: #9ca3af;
    --green: #16a34a; --green-bg: #dcfce7; --green-border: #86efac;
    --red:   #dc2626; --red-bg: #fee2e2; --red-border: #fca5a5;
    --blue:  #1d4ed8; --blue-bg: #dbeafe; --blue-border: #93c5fd;
    --amber-bg: #fef9c3; --amber-border: #fde047; --amber-text: #713f12;
    --radius: 10px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 60px; }}

  /* header */
  .header {{ margin-bottom: 28px; }}
  .header h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .header p  {{ color: var(--muted); font-size: 13px; }}

  /* metric cards row */
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }}
  .mc {{ background: var(--surface2); border-radius: var(--radius); padding: 14px 16px; }}
  .mc .label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: 4px; }}
  .mc .value {{ font-size: 24px; font-weight: 600; }}
  .mc .sub   {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
  .mc.highlight {{ border-left: 3px solid #3b82f6; }}

  /* two-col layout */
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }}

  /* section title */
  .sec {{ font-size: 11px; font-weight: 600; color: var(--muted);
          text-transform: uppercase; letter-spacing: .06em; margin-bottom: 12px; }}

  /* cluster rows */
  .cluster-row {{ border: 1px solid var(--border); border-radius: var(--radius);
                  padding: 14px 16px; margin-bottom: 10px; background: var(--surface); }}
  .cluster-row.active {{ border: 2px solid #3b82f6; background: #f0f7ff; }}
  .cluster-row.signal {{ border-color: var(--green-border); }}
  .cluster-row.warn   {{ border-color: var(--red-border); }}
  .cluster-top {{ display: flex; justify-content: space-between;
                  align-items: flex-start; margin-bottom: 10px; }}
  .cluster-name {{ font-size: 13px; font-weight: 600; }}
  .cluster-sub  {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
  .cluster-next {{ font-size: 11px; color: var(--muted); margin-top: 6px; }}

  /* probability bar */
  .bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .bar-label {{ font-size: 11px; color: var(--muted); min-width: 34px; }}
  .bar-bg {{ flex: 1; background: var(--border); border-radius: 4px;
             height: 7px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width .4s; }}
  .bar-ci {{ font-size: 10px; color: var(--hint); white-space: nowrap; }}

  /* chips */
  .chip {{ display: inline-block; font-size: 11px; padding: 3px 10px;
           border-radius: 20px; font-weight: 600; white-space: nowrap; }}
  .chip-up   {{ background: var(--green-bg); color: var(--green); }}
  .chip-down {{ background: var(--red-bg);   color: var(--red); }}
  .chip-neu  {{ background: var(--surface2); color: var(--muted);
                border: 1px solid var(--border2); }}
  .chip-today{{ background: var(--blue-bg); color: var(--blue);
                border: 1px solid var(--blue-border); font-size: 11px;
                padding: 2px 8px; border-radius: 4px; }}
  .sig-yes {{ display: inline-block; font-size: 10px; padding: 2px 6px;
              border-radius: 4px; background: var(--blue-bg); color: var(--blue);
              margin-left: 4px; font-weight: 600; }}
  .sig-no  {{ display: inline-block; font-size: 10px; padding: 2px 6px;
              border-radius: 4px; background: var(--surface2); color: var(--hint);
              margin-left: 4px; }}

  /* feature table */
  .feat-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .feat-table td {{ padding: 6px 0; border-bottom: 1px solid var(--border); }}
  .feat-table tr:last-child td {{ border-bottom: none; }}
  .feat-table .fn {{ color: var(--muted); }}
  .feat-table .fv {{ font-weight: 600; text-align: right; }}

  /* card */
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: var(--radius); padding: 16px 20px; }}
  .card.green-card {{ border-color: var(--green-border); }}
  .card.red-card   {{ border-color: var(--red-border); }}
  .card.amber-card {{ border-color: var(--amber-border);
                      background: var(--amber-bg); }}

  /* big signal box */
  .signal-box {{ padding: 14px; background: var(--surface2);
                 border-radius: 8px; margin-top: 12px; text-align: center; }}
  .signal-box .sig-val {{ font-size: 26px; font-weight: 700; }}
  .signal-box .sig-meta {{ font-size: 11px; color: var(--muted); margin-top: 3px; }}

  /* top signals table */
  .signals-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .signals-table th {{ text-align: left; color: var(--muted); font-weight: 600;
                       font-size: 11px; padding: 6px 8px; border-bottom: 2px solid var(--border); }}
  .signals-table td {{ padding: 7px 8px; border-bottom: 1px solid var(--border); }}
  .signals-table tr:last-child td {{ border-bottom: none; }}
  .signals-table tr:hover {{ background: var(--surface2); }}
  .badge {{ display: inline-block; font-size: 10px; padding: 1px 6px;
            border-radius: 3px; font-weight: 600; }}
  .badge-sig {{ background: var(--blue-bg); color: var(--blue); }}
  .badge-ns  {{ background: var(--surface2); color: var(--hint); }}

  /* full-width sections */
  .full {{ margin-bottom: 28px; }}
  .chart-wrap {{ position: relative; width: 100%; height: 200px; }}

  /* today alert */
  .alert {{ padding: 10px 14px; border-radius: 8px; font-size: 12px;
            margin-top: 10px; line-height: 1.6; }}
  .alert-amber {{ background: var(--amber-bg); border: 1px solid var(--amber-border);
                  color: var(--amber-text); }}
  .alert-green {{ background: var(--green-bg); border: 1px solid var(--green-border);
                  color: var(--green); }}

  /* footer */
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
             font-size: 11px; color: var(--hint); line-height: 1.7; }}

  @media (max-width: 700px) {{
    .metrics {{ grid-template-columns: 1fr 1fr; }}
    .grid2   {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <h1>SPY — Análisis Predictivo de Tipos de Día</h1>
    <p>Generado el {generated_at} · Datos: {date_range} · {N} días de trading</p>
  </div>

  <!-- METRIC CARDS -->
  <div class="metrics">
    <div class="mc">
      <div class="label">Días analizados</div>
      <div class="value">{N}</div>
      <div class="sub">{date_range_short}</div>
    </div>
    <div class="mc">
      <div class="label">Base rate (sube)</div>
      <div class="value">{base_rate_pct}%</div>
      <div class="sub">sin filtro</div>
    </div>
    <div class="mc">
      <div class="label">Señal más fuerte</div>
      <div class="value">{best_signal_pct}%</div>
      <div class="sub">{best_signal_name}</div>
    </div>
    <div class="mc highlight">
      <div class="label">Último día analizado</div>
      <div class="value" style="font-size:17px;">{today_cat_short}</div>
      <div class="sub">{today_p_pct}% prob. sube · {today_date}</div>
    </div>
  </div>

  <!-- MAIN GRID: categorías + hoy -->
  <div class="grid2">

    <!-- Categorías -->
    <div>
      <div class="sec">Categorías de días</div>
      {clusters_html}
    </div>

    <!-- Panel derecho -->
    <div style="display:flex; flex-direction:column; gap:16px;">

      <!-- Señal combinada -->
      <div class="card green-card">
        <div class="sec" style="color:var(--green);">★ Señal combinada más poderosa</div>
        <div style="font-size:14px; font-weight:600; margin-bottom:6px;">{combo_title}</div>
        <div style="font-size:12px; color:var(--muted); margin-bottom:12px;">{combo_desc}</div>
        <div style="display:flex; gap:12px; padding:12px; background:var(--surface2); border-radius:8px; align-items:center;">
          <div style="text-align:center; min-width:70px;">
            <div style="font-size:28px; font-weight:700; color:var(--green);">{combo_p_pct}%</div>
            <div style="font-size:11px; color:var(--muted);">prob. alcista</div>
          </div>
          <div style="font-size:12px; color:var(--muted); line-height:1.7;">
            IC 90%: [{combo_ci_lo}% – {combo_ci_hi}%]<br>
            p-valor: <strong style="color:var(--blue);">{combo_pval}</strong>
            {combo_sig_badge}<br>
            Ret. medio siguiente: <strong style="color:var(--green);">{combo_next_ret:+.2f}%</strong>
          </div>
        </div>
        <div style="font-size:11px; color:var(--muted); margin-top:10px; line-height:1.6;">
          {combo_interpretation}
        </div>
      </div>

      <!-- Hoy -->
      <div class="card" style="border-color: #3b82f6;">
        <div class="sec">Último día: {today_date}</div>
        <table class="feat-table">
          {today_features_html}
        </table>
        <div class="signal-box">
          <div style="font-size:11px; color:var(--muted);">Señal para el día siguiente</div>
          <div class="sig-val" style="color:{today_color};">{today_p_pct}% prob. alcista</div>
          <div class="sig-meta">Categoría: {today_cat} · base rate {base_rate_pct}%</div>
        </div>
        {today_alert_html}
      </div>

    </div>
  </div>

  <!-- TOP SEÑALES SIMPLES -->
  <div class="full">
    <div class="sec">Top señales de una sola variable</div>
    <div class="card">
      <table class="signals-table">
        <thead>
          <tr>
            <th>Condición</th><th>Días</th><th>% del tiempo</th>
            <th>P(sube)</th><th>IC 90%</th><th>p-valor</th><th>Edge vs base</th>
          </tr>
        </thead>
        <tbody>
          {top_single_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- TOP SEÑALES COMBINADAS -->
  <div class="full">
    <div class="sec">Top señales combinadas (2 variables)</div>
    <div class="card">
      <table class="signals-table">
        <thead>
          <tr>
            <th>Condición</th><th>Días</th><th>% del tiempo</th>
            <th>P(sube)</th><th>IC 90%</th><th>p-valor</th><th>Edge vs base</th>
          </tr>
        </thead>
        <tbody>
          {top_two_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- KMEANS SCORE CHART -->
  <div class="full">
    <div class="sec">Score k-means por número de clusters</div>
    <div class="card">
      <div class="chart-wrap">
        <canvas id="nChart"></canvas>
      </div>
      <p style="font-size:11px; color:var(--hint); margin-top:10px; line-height:1.6;">
        Score = 35% cohesión interna (silhouette) + 65% edge predictivo ponderado por tamaño.
        Con suficientes días, el k-means converge en pocos clusters de bajo edge.
        La segmentación por reglas interpretables (arriba) resulta más accionable.
      </p>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <strong>Metodología:</strong> segmentación por reglas sobre 10 features horarios
    (retorno, rango, gap, posición del cierre, mañana/tarde, primera/última hora, volumen relativo y tendencia de volumen).
    IC = intervalo de confianza Wilson 90%. p-valor = test-t bilateral vs H₀: p=0.5.
    Solo se consideran señales estadísticamente relevantes (p&lt;0.10) para decisiones.
    <br><br>
    <strong>Aviso:</strong> este análisis es puramente estadístico sobre datos históricos.
    No constituye asesoramiento financiero. Los patrones pasados no garantizan resultados futuros.
    Usar como señal complementaria, nunca como sistema de trading autónomo.
  </div>

</div>

<script>
const nData = {kmeans_json};
new Chart(document.getElementById('nChart'), {{
  type: 'bar',
  data: {{
    labels: nData.map(d => 'N=' + d.n),
    datasets: [{{
      data: nData.map(d => d.score),
      backgroundColor: nData.map(d => d.n === {best_n} ? '#3b82f6' : '#d1d5db'),
      borderRadius: 4, borderSkipped: false,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: c => ' Score: ' + c.parsed.y.toFixed(3) }} }} }},
    scales: {{
      x: {{ grid: {{ display: false }},
            ticks: {{ font: {{ size: 11 }}, color: '#9ca3af' }} }},
      y: {{ min: 0,
            grid: {{ color: 'rgba(0,0,0,0.06)' }},
            ticks: {{ font: {{ size: 11 }}, color: '#9ca3af',
                      callback: v => v.toFixed(3) }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""


def bar_color(p_up, base):
    if p_up >= base + 0.07:
        return "#22c55e"
    if p_up <= base - 0.07:
        return "#ef4444"
    return "#94a3b8"


def chip_html(p_up, base):
    pct = round(p_up * 100, 1)
    if p_up >= base + 0.07:
        return f'<span class="chip chip-up">{pct}% ↑</span>'
    if p_up <= base - 0.07:
        return f'<span class="chip chip-down">{pct}% ↑</span>'
    return f'<span class="chip chip-neu">{pct}% ↑</span>'


def sig_badge(pval):
    if pval < 0.05:
        return f'<span class="sig-yes">★ p={pval}</span>'
    if pval < 0.10:
        return f'<span class="sig-yes" style="background:#fef9c3;color:#92400e;">~ p={pval}</span>'
    return f'<span class="sig-no">p={pval}</span>'


def build_cluster_html(cat_summary, today_cat, base):
    descs = {
        "A: Alcista fuerte":       "Cierra en altos (>70%) · retorno >+0.3%",
        "B: Alcista con tarde fort":"Cierra en altos (>65%) · tarde alcista >+0.2%",
        "C: Lateral":              "Cierra en zona media (35–65% del rango)",
        "D: Bajista tarde débil":  "Cierra en bajos (<35%) · tarde bajista <-0.2%",
        "E: Capitulación":         "Cierra en bajos (<30%) · retorno <-0.3%",
        "F: Resto":                "Días que no encajan en las categorías anteriores",
    }
    html = ""
    for r in cat_summary:
        name = r["name"]
        is_today = (name == today_cat)
        is_signal = r["p_up"] >= base + 0.07
        is_warn   = r["p_up"] <= base - 0.07

        cls = "cluster-row"
        if is_today:   cls += " active"
        elif is_signal: cls += " signal"
        elif is_warn:   cls += " warn"

        today_badge = ' <span class="chip-today">◀ último día</span>' if is_today else ""
        pct = round(r["p_up"] * 100, 1)

        html += f"""
        <div class="{cls}">
          <div class="cluster-top">
            <div>
              <span class="cluster-name">{name}{today_badge}</span>
              {sig_badge(r["pval"])}
              <div class="cluster-sub">{descs.get(name, "")} · {r["n"]} días ({r["pct_days"]}%)</div>
            </div>
            {chip_html(r["p_up"], base)}
          </div>
          <div class="bar-wrap">
            <span class="bar-label">{pct}%</span>
            <div class="bar-bg">
              <div class="bar-fill" style="width:{pct}%;background:{bar_color(r['p_up'], base)};"></div>
            </div>
            <span class="bar-ci">[{round(r['ci_lo']*100)}–{round(r['ci_hi']*100)}%]</span>
          </div>
          <div class="cluster-next">
            Ret. medio día siguiente: <strong style="color:{'#16a34a' if r['next_ret_mean']>0 else '#dc2626'};">{r['next_ret_mean']:+.3f}%</strong>
          </div>
        </div>"""
    return html


def build_signals_rows(records, base):
    rows = ""
    for r in records:
        pct = round(r["p_up"] * 100, 1)
        direction = "↑" if r["p_up"] > base else "↓"
        color = "#16a34a" if r["p_up"] >= base + 0.05 else ("#dc2626" if r["p_up"] <= base - 0.05 else "#374151")
        sig_cls = "badge-sig" if r["pval"] < 0.10 else "badge-ns"
        rows += f"""
        <tr>
          <td>{r['name']}</td>
          <td>{r['n']}</td>
          <td>{r['pct_days']}%</td>
          <td style="font-weight:600;color:{color};">{pct}% {direction}</td>
          <td>[{round(r['ci_lo']*100)}–{round(r['ci_hi']*100)}%]</td>
          <td><span class="badge {sig_cls}">{r['pval']}</span></td>
          <td style="font-weight:600;">{round(r['edge']*100,1)} pp</td>
        </tr>"""
    return rows


def build_today_features(features_dict):
    labels = {
        "ret":           "Retorno del día",
        "range_pct":     "Rango intradiario",
        "gap":           "Gap de apertura",
        "close_pos":     "Posición del cierre",
        "morning_ret":   "Retorno mañana",
        "afternoon_ret": "Retorno tarde",
        "vol_rel":       "Volumen relativo",
    }
    rows = ""
    for key, label in labels.items():
        val = features_dict.get(key, 0)
        if key == "close_pos":
            display = f"{round(val*100, 1)}% del rango"
            color = ""
        elif key == "vol_rel":
            display = f"{val:.2f}x promedio"
            color = ""
        else:
            color = f"color:{'#16a34a' if val > 0 else '#dc2626'};"
            display = f"{val:+.3f}%"
        rows += f'<tr><td class="fn">{label}</td><td class="fv" style="{color}">{display}</td></tr>'
    return rows


def generate_html(results, today, csv_name):
    base     = results["base_rate"]
    combo    = results["combo"] or {}
    best_cat = max(results["cat_summary"], key=lambda r: r["edge"], default={})
    best_n   = max(results["kmeans_scores"], key=lambda r: r["score"])["n"]

    # today color
    p_today = today["p_up"]
    if p_today >= base + 0.07:
        today_color = "#16a34a"
    elif p_today <= base - 0.07:
        today_color = "#dc2626"
    else:
        today_color = "#374151"

    # combo details
    combo_p_pct   = round(combo.get("p_up", 0) * 100, 1) if combo else 0
    combo_n       = combo.get("n", 0)
    combo_ci_lo   = round((combo.get("ci_lo", 0)) * 100, 0) if combo else 0
    combo_ci_hi   = round((combo.get("ci_hi", 0)) * 100, 0) if combo else 0
    combo_pval    = combo.get("pval", 1) if combo else 1
    combo_next    = combo.get("next_ret_mean", 0) if combo else 0
    ret_thr       = results["ret_thr"]
    gap_thr       = results["gap_thr"]

    # today alert
    today_alert = ""
    if today.get("combo_active"):
        today_alert = f"""<div class="alert alert-green">
          ★ El último día activa la señal combinada más poderosa (ret &lt; {ret_thr}% y gap &gt; {gap_thr}%).
          Probabilidad estimada de suba: <strong>{combo_p_pct}%</strong>.
        </div>"""
    elif today["features"]["ret"] > ret_thr and today["features"]["gap"] > gap_thr:
        today_alert = f"""<div class="alert alert-amber">
          ⚠ El día se acerca a la señal combinada (ret={today['features']['ret']:+.2f}% / umbral {ret_thr}%
          · gap={today['features']['gap']:+.2f}% / umbral {gap_thr}%). No llega al umbral pero el contexto es similar.
        </div>"""

    # date range short
    dr = results["date_range"]
    parts = dr.split(" → ")
    date_range_short = f"{parts[0][:7]} – {parts[1][:7]}" if len(parts) == 2 else dr

    return HTML_TEMPLATE.format(
        date_range        = results["date_range"],
        date_range_short  = date_range_short,
        generated_at      = datetime.now().strftime("%d/%m/%Y %H:%M"),
        N                 = results["N"],
        base_rate_pct     = round(base * 100, 1),
        best_signal_pct   = round(best_cat.get("p_up", 0) * 100, 1) if best_cat else "—",
        best_signal_name  = (best_cat.get("name", "")[:22] + "…") if best_cat else "—",
        today_cat_short   = today["cat"].split(":")[0] + ": " + today["cat"].split(":")[1][:12] if ":" in today["cat"] else today["cat"][:16],
        today_p_pct       = round(p_today * 100, 1),
        today_date        = today["date"],
        today_cat         = today["cat"],
        today_color       = today_color,
        clusters_html     = build_cluster_html(results["cat_summary"], today["cat"], base),
        combo_title       = f"Retorno < {ret_thr}% y Gap de apertura > {gap_thr}%",
        combo_desc        = f"{combo_n} días ({round(combo_n/results['N']*100,1)}% del tiempo). "
                            f"El mercado baja durante el día pero abre con gap positivo.",
        combo_p_pct       = combo_p_pct,
        combo_ci_lo       = int(combo_ci_lo),
        combo_ci_hi       = int(combo_ci_hi),
        combo_pval        = combo_pval,
        combo_sig_badge   = "(estadísticamente significativo ✓)" if combo_pval < 0.05 else "(tendencia, no significativo)",
        combo_next_ret    = combo_next,
        combo_interpretation = (
            "La presión vendedora intradía se agota cuando los institucionales siguen "
            "abriendo con gap positivo. Señal de demanda subyacente → tendencia a rebote al día siguiente."
        ),
        today_features_html = build_today_features(today["features"]),
        today_alert_html    = today_alert,
        top_single_rows     = build_signals_rows(results["top_single"], base),
        top_two_rows        = build_signals_rows(results["top_two"], base),
        kmeans_json         = json.dumps(results["kmeans_scores"]),
        best_n              = best_n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analiza datos horarios del SPY y genera un reporte HTML predictivo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python spy_analysis.py SPY_1h.csv
  python spy_analysis.py SPY_1h.csv --output mi_reporte.html
  python spy_analysis.py SPY_1h.csv --umbral-ret -0.5 --umbral-gap 0.2
        """
    )
    parser.add_argument("csv",           help="Ruta al archivo CSV de datos horarios del SPY")
    parser.add_argument("--output", "-o",default=None,
                        help="Nombre del HTML de salida (default: spy_report_YYYYMMDD.html)")
    parser.add_argument("--umbral-ret",  type=float, default=-0.3,
                        help="Umbral de retorno para señal combinada (default: -0.3)")
    parser.add_argument("--umbral-gap",  type=float, default=0.15,
                        help="Umbral de gap para señal combinada (default: 0.15)")
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"ERROR: No se encontró el archivo '{csv_path}'")
        sys.exit(1)

    cfg = {"umbral_ret": args.umbral_ret, "umbral_gap": args.umbral_gap}

    output_path = args.output or f"spy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    print(f"Cargando datos: {csv_path}")
    daily, daily_clean, features = load_and_build_daily(csv_path)
    print(f"  → {len(daily_clean)} días de trading disponibles")

    print("Ejecutando análisis estadístico...")
    results = run_analysis(daily_clean, features, cfg)
    print(f"  → Base rate: {results['base_rate']:.1%} | Señal más fuerte: {max(results['cat_summary'], key=lambda r: r['edge'])['name']}")

    print("Clasificando último día...")
    today = classify_today(daily, daily_clean, cfg)
    print(f"  → {today['date']} | Categoría: {today['cat']} | P(sube): {today['p_up']:.1%}")

    print("Generando HTML...")
    html = generate_html(results, today, csv_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Reporte guardado en: {output_path}")
    print(f"  Abrilo con cualquier navegador (Chrome, Firefox, Safari, Edge).")


if __name__ == "__main__":
    main()
