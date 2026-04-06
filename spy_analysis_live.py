"""
SPY Live Day Analyzer
======================
Toma dos CSVs acumulados del SPY (1h y 15m, incluyendo datos parciales del día actual)
y genera un reporte HTML con:
  1. Caracterización completa del día actual
  2. Predicción de cómo va a cerrar hoy
  3. Predicción de si mañana sube o baja
  4. Señales condicionales históricas y días análogos

Uso:
    python spy_live_analysis.py datos_1h.csv datos_15m.csv
    python spy_live_analysis.py datos_1h.csv datos_15m.csv --output reporte.html

Dependencias:
    pip install pandas numpy scipy
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = df.index.date
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTRUCCIÓN DE FEATURES DIARIAS HISTÓRICAS
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_history(df1h: pd.DataFrame, today_date) -> pd.DataFrame:
    """Construye features de cada día histórico completo (excluye hoy)."""
    days = []
    for date, g in df1h.groupby("date"):
        if date == today_date:
            continue
        g = g.sort_index()
        if len(g) < 4:
            continue
        o = float(g["open"].iloc[0])
        c = float(g["close"].iloc[-1])
        h = float(g["high"].max())
        l = float(g["low"].min())
        mid = len(g) // 2
        days.append({
            "date":        date,
            "open":        o,
            "close":       c,
            "high":        h,
            "low":         l,
            "volume":      float(g["volume"].sum()),
            "ret":         (c - o) / o * 100,
            "range_pct":   (h - l) / o * 100,
            "morning_ret": (float(g["close"].iloc[mid - 1]) - o) / o * 100,
            "close_pos":   (c - l) / (h - l) if h != l else 0.5,
        })

    daily = pd.DataFrame(days).set_index("date")
    daily["gap"]      = (daily["open"] - daily["close"].shift(1)) / daily["close"].shift(1) * 100
    daily["vol_rel"]  = daily["volume"] / daily["volume"].rolling(20, min_periods=5).mean()
    daily["next_ret"] = daily["ret"].shift(-1)
    daily["up"]       = (daily["next_ret"] > 0).astype(float)
    daily = daily.dropna(subset=["ret", "gap", "vol_rel", "next_ret"]).copy()
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURES DEL DÍA ACTUAL (PARCIAL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_today_features(df1h: pd.DataFrame, df15: pd.DataFrame,
                           today_date, daily: pd.DataFrame) -> dict:
    t1h = df1h[df1h["date"] == today_date].sort_index()
    t15 = df15[df15["date"] == today_date].sort_index()

    o    = float(t1h["open"].iloc[0])
    c    = float(t15["close"].iloc[-1])
    h    = float(t1h["high"].max())
    l    = float(t1h["low"].min())
    prev = float(daily["close"].iloc[-1])
    mid  = max(1, len(t1h) // 2)

    ret_now   = (c - o) / o * 100
    range_now = (h - l) / o * 100
    cp_now    = (c - l) / (h - l) if h != l else 0.5
    gap       = (o - prev) / prev * 100
    morn      = (float(t1h["close"].iloc[mid - 1]) - o) / o * 100

    fh_close = float(t15["close"].iloc[min(3, len(t15) - 1)])
    fh_open  = float(t15["open"].iloc[0])
    fh_ret   = (fh_close - fh_open) / fh_open * 100

    n8 = min(8, len(t15) // 2)
    early = float(t15["close"].head(n8).mean())
    late  = float(t15["close"].tail(n8).mean())
    trend = (late - early) / early * 100 if early != 0 else 0.0

    avg_vol = daily["volume"].mean() / 7  # per-bar average
    vol_rel_proj = float(t1h["volume"].sum()) / (avg_vol * len(t1h)) if avg_vol > 0 else 1.0

    # Bin labels
    cp_bin    = "cp_bajo" if cp_now < .35 else ("cp_medio" if cp_now < .55 else ("cp_medio_alto" if cp_now < .75 else "cp_alto"))
    gap_bin   = "gap_bajo" if gap < -.2 else ("gap_alto" if gap > .2 else "gap_neutro")
    ret_bin   = "ret_neg" if ret_now < -.3 else ("ret_pos" if ret_now > .3 else "ret_plano")
    intra_bin = "bajista" if trend < -.1 else ("alcista" if trend > .1 else "neutral")

    return {
        "open": round(o, 2), "current_close": round(c, 2),
        "high": round(h, 2), "low": round(l, 2), "prev_close": round(prev, 2),
        "ret_so_far": round(ret_now, 4), "range_so_far": round(range_now, 4),
        "close_pos_now": round(cp_now, 4), "gap_today": round(gap, 4),
        "morning_ret": round(morn, 4), "first_hour_ret": round(fh_ret, 4),
        "intra_trend": round(trend, 4), "vol_rel_proj": round(vol_rel_proj, 3),
        "n_bars_1h": len(t1h), "n_bars_15m": len(t15),
        "cp_bin": cp_bin, "gap_bin": gap_bin, "ret_bin": ret_bin, "intra_bin": intra_bin,
        "as_of": str(t15.index[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICCIÓN: CIERRE DE HOY
# ─────────────────────────────────────────────────────────────────────────────

def predict_today_close(df1h: pd.DataFrame, today_date, today: dict) -> dict:
    """Encuentra días históricos con retorno parcial similar a la misma barra."""
    pvf = []
    for date, g in df1h.groupby("date"):
        if date == today_date:
            continue
        g = g.sort_index()
        if len(g) < 6:
            continue
        op  = float(g["open"].iloc[0])
        r5  = (float(g["close"].iloc[4]) - op) / op * 100
        rf  = (float(g["close"].iloc[-1]) - op) / op * 100
        pvf.append({
            "ret_partial":  r5,
            "ret_final":    rf,
            "up_from_here": 1 if rf > r5 else 0,
        })

    pvf_df = pd.DataFrame(pvf)
    tol    = 0.3
    ret    = today["ret_so_far"]
    sim    = pvf_df[(pvf_df["ret_partial"] >= ret - tol) & (pvf_df["ret_partial"] <= ret + tol)]

    p_up  = float(sim["up_from_here"].mean()) if len(sim) >= 5 else 0.5
    delta = float((sim["ret_final"] - sim["ret_partial"]).mean()) if len(sim) >= 5 else 0.0
    proj  = today["current_close"] * (1 + delta / 100)

    # Histogram: ret_partial bins → P(up from here)
    hist = []
    for lo in np.arange(-2.5, 2.5, 0.5):
        hi  = lo + 0.5
        sub = pvf_df[(pvf_df["ret_partial"] >= lo) & (pvf_df["ret_partial"] < hi)]
        hist.append({
            "label": f"{lo:+.1f}",
            "n":     int(len(sub)),
            "p_up":  round(float(sub["up_from_here"].mean()) * 100, 1) if len(sub) > 2 else 0,
        })

    return {
        "similar_n":      int(len(sim)),
        "p_up_from_here": round(p_up, 4),
        "delta_mean":     round(delta, 4),
        "proj_close":     round(proj, 2),
        "pvf_hist":       hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICCIÓN: MAÑANA
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(p, n, conf=0.90):
    z = norm.ppf((1 + conf) / 2)
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return round(c - m, 3), round(c + m, 3)


def _analyze(mask, daily):
    g = daily[mask]
    if len(g) < 6:
        return None
    p  = float(g["up"].mean())
    n  = len(g)
    ci = _wilson_ci(p, n)
    _, pval = ttest_1samp(g["up"].values, 0.5)
    return {
        "n": n, "p_up": round(p, 4),
        "ci_lo": ci[0], "ci_hi": ci[1],
        "pval": round(pval, 3),
        "next_ret_mean": round(float(g["next_ret"].mean()), 4),
    }


def predict_tomorrow(daily: pd.DataFrame, today: dict) -> dict:
    """k-NN top-20 por similitud multivariada + señales condicionales."""
    feat_cols = ["ret", "range_pct", "gap", "close_pos", "morning_ret"]
    tv   = np.array([today["ret_so_far"], today["range_so_far"],
                     today["gap_today"], today["close_pos_now"], today["morning_ret"]])
    stds = np.array([float(daily[f].std()) for f in feat_cols])
    stds[stds == 0] = 1
    dists = np.sqrt(((daily[feat_cols].values - tv) / stds) ** 2).sum(axis=1)
    daily["dist"] = dists
    top20 = daily.nsmallest(20, "dist")

    p20  = float(top20["up"].mean())
    ci20 = _wilson_ci(p20, 20)
    nr20 = float(top20["next_ret"].mean())

    top20_list = [
        {"date": str(r.name), "ret": round(r["ret"], 3), "gap": round(r["gap"], 3),
         "cp": round(r["close_pos"], 3), "up": int(r["up"]),
         "nr": round(r["next_ret"], 3), "dist": round(r["dist"], 3)}
        for _, r in top20.iterrows()
    ]

    # Conditional signals
    signals = {}
    for lo, hi, k in [(0, .35, "cp_bajo"), (.35, .55, "cp_medio"), (.55, .75, "cp_medio_alto"), (.75, 1.01, "cp_alto")]:
        r = _analyze((daily["close_pos"] >= lo) & (daily["close_pos"] < hi), daily)
        if r: signals[k] = r
    for lo, hi, k in [(-99, -.2, "gap_bajo"), (-.2, .2, "gap_neutro"), (.2, 99, "gap_alto")]:
        r = _analyze((daily["gap"] >= lo) & (daily["gap"] < hi), daily)
        if r: signals[k] = r
    for lo, hi, k in [(-99, -.3, "ret_neg"), (-.3, .3, "ret_plano"), (.3, 99, "ret_pos")]:
        r = _analyze((daily["ret"] >= lo) & (daily["ret"] < hi), daily)
        if r: signals[k] = r

    return {
        "p_up_top20":   round(p20, 4),
        "ci_lo":        ci20[0],
        "ci_hi":        ci20[1],
        "next_ret_mean": round(nr20, 4),
        "top20":        top20_list,
        "signals":      signals,
        "today_cp_sig": signals.get(today["cp_bin"]),
        "today_gap_sig": signals.get(today["gap_bin"]),
        "today_ret_sig": signals.get(today["ret_bin"]),
    }


def predict_intra_signal(df15: pd.DataFrame, daily: pd.DataFrame, today_date, today: dict) -> dict:
    """Señal de tendencia intradiaria (late vs early promedio)."""
    rows = []
    for date, g in df15.groupby("date"):
        if date == today_date:
            continue
        g = g.sort_index()
        if len(g) < 12:
            continue
        n8 = min(8, len(g) // 2)
        e  = float(g["close"].head(n8).mean())
        la = float(g["close"].tail(n8).mean())
        rows.append({"date": date, "intra_trend": (la - e) / e * 100})

    d15f  = pd.DataFrame(rows).set_index("date").join(daily[["up", "next_ret"]]).dropna()
    intra = {}
    for lo, hi, k in [(-99, -.1, "bajista"), (-.1, .1, "neutral"), (.1, 99, "alcista")]:
        g2 = d15f[(d15f["intra_trend"] >= lo) & (d15f["intra_trend"] < hi)]
        if len(g2) >= 6:
            p2  = float(g2["up"].mean())
            ci2 = _wilson_ci(p2, len(g2))
            _, pv = ttest_1samp(g2["up"].values, 0.5)
            intra[k] = {"n": len(g2), "p_up": round(p2, 4),
                        "ci_lo": ci2[0], "ci_hi": ci2[1], "pval": round(pv, 3)}

    return {"intra_signals": intra, "today_intra_sig": intra.get(today["intra_bin"])}


# ─────────────────────────────────────────────────────────────────────────────
# 6. GENERACIÓN DEL HTML
# ─────────────────────────────────────────────────────────────────────────────

def pct(v):
    return f"{v * 100:.1f}%"

def signed(v, decimals=3):
    return f"{v:+.{decimals}f}%"

def color_val(v, positive="#16a34a", negative="#dc2626", neutral="#888"):
    if v > 0.01:   return positive
    if v < -0.01:  return negative
    return neutral


def _chip(p, base):
    pct_val = round(p * 100, 1)
    if p >= base + 0.05:
        return f'<span style="background:#dcfce7;color:#166534;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500">{pct_val}% ↑</span>'
    if p <= base - 0.05:
        return f'<span style="background:#fee2e2;color:#991b1b;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500">{pct_val}% ↑</span>'
    return f'<span style="background:#f3f4f6;color:#6b7280;font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500">{pct_val}% ↑</span>'


def _bar_row(label, sig, base):
    if not sig:
        return ""
    p   = sig["p_up"]
    pv  = sig["pval"]
    ci  = f"[{round(sig['ci_lo']*100)}–{round(sig['ci_hi']*100)}%]"
    pv_badge = f'<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:#dbeafe;color:#1e40af;margin-left:4px">p={pv}</span>' if pv < 0.10 else f'<span style="font-size:10px;padding:2px 6px;border-radius:4px;background:#f3f4f6;color:#9ca3af;margin-left:4px">p={pv}</span>'
    bar_color = "#22c55e" if p >= base + 0.05 else ("#ef4444" if p <= base - 0.05 else "#94a3b8")
    return f"""
    <div style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
        <span style="font-size:13px;color:#374151">{label}</span>
        {_chip(p, base)}{pv_badge}
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:11px;color:#9ca3af;min-width:32px">{round(p*100,1)}%</span>
        <div style="flex:1;background:#e5e7eb;border-radius:3px;height:6px;overflow:hidden">
          <div style="width:{round(p*100,1)}%;height:100%;border-radius:3px;background:{bar_color}"></div>
        </div>
        <span style="font-size:10px;color:#9ca3af">{ci}</span>
      </div>
    </div>"""


def _top20_rows(top20):
    rows = ""
    for d in top20:
        up_color = "#16a34a" if d["up"] else "#dc2626"
        up_text  = "Sube" if d["up"] else "Baja"
        nr_color = "#16a34a" if d["nr"] > 0 else "#dc2626"
        rows += f"""<tr>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb">{d['date']}</td>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb;text-align:center">{d['ret']:+.2f}%</td>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb;text-align:center">{d['gap']:+.2f}%</td>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb;text-align:center">{d['cp']:.2f}</td>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb;text-align:center;color:{up_color}">{up_text}</td>
          <td style="padding:4px 8px;border-bottom:0.5px solid #e5e7eb;text-align:right;color:{nr_color}">{d['nr']:+.2f}%</td>
        </tr>"""
    return rows


def _verdict_today(p_up):
    if p_up > 0.54:
        return '<div style="padding:12px;border-radius:8px;background:#dcfce7;border:1px solid #86efac;text-align:center"><div style="font-size:12px;color:#166534">Cierre de hoy</div><div style="font-size:20px;font-weight:500;color:#166534">Sesgo alcista</div></div>'
    if p_up < 0.46:
        return '<div style="padding:12px;border-radius:8px;background:#fee2e2;border:1px solid #fca5a5;text-align:center"><div style="font-size:12px;color:#991b1b">Cierre de hoy</div><div style="font-size:20px;font-weight:500;color:#991b1b">Sesgo bajista</div></div>'
    return '<div style="padding:12px;border-radius:8px;background:#f3f4f6;border:0.5px solid #d1d5db;text-align:center"><div style="font-size:12px;color:#6b7280">Cierre de hoy</div><div style="font-size:20px;font-weight:500;color:#374151">Sin sesgo — 50/50</div></div>'


def _verdict_tomorrow(p_up, ci_lo, ci_hi):
    pct_v = round(p_up * 100, 1)
    ci_str = f"IC 90%: [{round(ci_lo*100)}%–{round(ci_hi*100)}%]"
    if p_up > 0.57:
        return f'<div style="padding:12px;border-radius:8px;background:#dcfce7;border:1px solid #86efac;text-align:center"><div style="font-size:12px;color:#166534">Mañana</div><div style="font-size:20px;font-weight:500;color:#166534">Sesgo alcista — {pct_v}%</div><div style="font-size:11px;color:#166534;margin-top:2px">{ci_str}</div></div>'
    if p_up < 0.43:
        return f'<div style="padding:12px;border-radius:8px;background:#fee2e2;border:1px solid #fca5a5;text-align:center"><div style="font-size:12px;color:#991b1b">Mañana</div><div style="font-size:20px;font-weight:500;color:#991b1b">Sesgo bajista — {pct_v}%</div><div style="font-size:11px;color:#991b1b;margin-top:2px">{ci_str}</div></div>'
    return f'<div style="padding:12px;border-radius:8px;background:#f3f4f6;border:0.5px solid #d1d5db;text-align:center"><div style="font-size:12px;color:#6b7280">Mañana</div><div style="font-size:20px;font-weight:500;color:#374151">Sin sesgo claro — {pct_v}%</div><div style="font-size:11px;color:#6b7280;margin-top:2px">{ci_str}</div></div>'


def generate_html(today_date, today, today_pred, tmrw, intra, base, total_hist):
    t   = today
    tp  = today_pred
    tm  = tmrw
    sig = tmrw["signals"]

    # Signals bar section
    cp_label   = {"cp_bajo": "Posición cierre baja (<35%)", "cp_medio": "Posición cierre media (35–55%)",
                  "cp_medio_alto": "Posición cierre media-alta (55–75%)", "cp_alto": "Posición cierre alta (>75%)"}
    gap_label  = {"gap_bajo": "Gap bajista (<-0.2%)", "gap_neutro": "Gap neutro (±0.2%)", "gap_alto": "Gap alcista (>+0.2%)"}
    ret_label  = {"ret_neg": "Retorno negativo (<-0.3%)", "ret_plano": "Retorno plano (±0.3%)", "ret_pos": "Retorno positivo (>+0.3%)"}
    intra_lbl  = {"bajista": "Tendencia intradiaria bajista", "neutral": "Tendencia intradiaria neutral", "alcista": "Tendencia intradiaria alcista"}

    signals_html = ""
    signals_html += _bar_row(cp_label.get(t["cp_bin"], t["cp_bin"]),   sig.get(t["cp_bin"]),   base)
    signals_html += _bar_row(gap_label.get(t["gap_bin"], t["gap_bin"]), sig.get(t["gap_bin"]),  base)
    signals_html += _bar_row(ret_label.get(t["ret_bin"], t["ret_bin"]), sig.get(t["ret_bin"]),  base)
    intra_sig_today = intra["intra_signals"].get(t["intra_bin"])
    signals_html += _bar_row(intra_lbl.get(t["intra_bin"], t["intra_bin"]), intra_sig_today, base)

    # Top5 similar days
    top5_rows = _top20_rows(tm["top20"][:5])

    # Hist chart data
    hist_labels = json.dumps([h["label"] for h in tp["pvf_hist"]])
    hist_pup    = json.dumps([h["p_up"] if h["p_up"] is not None else 0 for h in tp["pvf_hist"]])
    hist_n      = json.dumps([h["n"] for h in tp["pvf_hist"]])
    # Find today's bucket index
    today_bucket = 0
    for i, h in enumerate(tp["pvf_hist"]):
        try:
            lo = float(h["label"])
            if lo <= t["ret_so_far"] < lo + 0.5:
                today_bucket = i
                break
        except Exception:
            pass

    today_close_color = color_val(t["ret_so_far"])
    gap_color  = color_val(t["gap_today"])
    trend_color= color_val(t["intra_trend"])

    delta_color = "#dc2626" if tp["delta_mean"] < 0 else "#16a34a"
    proj_price  = tp["proj_close"]

    n_bars_expected_1h  = 7
    n_bars_expected_15m = 26

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SPY Live Analysis — {today_date}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f9fafb;color:#111827;font-size:14px;line-height:1.5}}
  .page{{max-width:1080px;margin:0 auto;padding:32px 20px 60px}}
  .header{{margin-bottom:24px}}
  .header h1{{font-size:21px;font-weight:700;margin-bottom:4px}}
  .header p{{color:#6b7280;font-size:13px}}
  .g4{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:18px}}
  .g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
  .mc{{background:#f3f4f6;border-radius:8px;padding:12px 14px}}
  .mc .lb{{font-size:11px;color:#6b7280;margin:0 0 3px;text-transform:uppercase;letter-spacing:.04em}}
  .mc .vl{{font-size:22px;font-weight:600;color:#111827;margin:0}}
  .mc .sb{{font-size:11px;color:#6b7280;margin:2px 0 0}}
  .card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 18px;margin-bottom:14px}}
  .card.green-border{{border-color:#86efac}}
  .sec{{font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin:0 0 10px}}
  .feat{{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:0.5px solid #f3f4f6;font-size:13px}}
  .feat:last-child{{border-bottom:none}}
  .fn{{color:#6b7280}}
  .fv{{font-weight:600}}
  .verdict{{padding:12px 16px;border-radius:8px;margin-top:10px;text-align:center}}
  .verdict-title{{font-size:12px;color:#6b7280}}
  .verdict-main{{font-size:20px;font-weight:600;margin-top:2px}}
  .verdict-sub{{font-size:11px;margin-top:2px}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{text-align:left;color:#6b7280;font-weight:600;font-size:11px;padding:6px 8px;border-bottom:2px solid #e5e7eb;background:#f9fafb}}
  td{{padding:4px 8px;border-bottom:0.5px solid #f3f4f6}}
  tr:hover td{{background:#f9fafb}}
  .footer{{margin-top:36px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:11px;color:#9ca3af;line-height:1.7}}
  @media(max-width:700px){{.g4{{grid-template-columns:1fr 1fr}}.g2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="page">

<div class="header">
  <h1>SPY — Análisis en vivo · {today_date}</h1>
  <p>Datos al {t['as_of']} ET &nbsp;·&nbsp; {total_hist} días históricos (abr 2023 – hoy) &nbsp;·&nbsp; Base rate sube: {round(base*100,1)}%</p>
</div>

<div class="g4">
  <div class="mc"><div class="lb">Apertura</div><div class="vl">${t['open']:.2f}</div><div class="sb">prev. ${t['prev_close']:.2f} · gap {t['gap_today']:+.2f}%</div></div>
  <div class="mc"><div class="lb">Precio actual</div><div class="vl" style="color:{today_close_color}">${t['current_close']:.2f}</div><div class="sb" style="color:{today_close_color}">{t['ret_so_far']:+.3f}% desde apertura</div></div>
  <div class="mc"><div class="lb">Rango del día</div><div class="vl">${(t['high']-t['low']):.2f}</div><div class="sb">${t['low']:.2f} – ${t['high']:.2f}</div></div>
  <div class="mc"><div class="lb">Progreso sesión</div><div class="vl">{t['n_bars_1h']}/{n_bars_expected_1h}</div><div class="sb">barras 1h · {t['n_bars_15m']}/{n_bars_expected_15m} barras 15m</div></div>
</div>

<div class="g2">

  <div>
    <div class="card">
      <div class="sec">Caracterización del día</div>
      <div class="feat"><span class="fn">Retorno hasta ahora</span><span class="fv" style="color:{today_close_color}">{t['ret_so_far']:+.3f}%</span></div>
      <div class="feat"><span class="fn">Gap de apertura</span><span class="fv" style="color:{gap_color}">{t['gap_today']:+.3f}% ({t['gap_bin'].replace('_',' ')})</span></div>
      <div class="feat"><span class="fn">Posición del precio en rango</span><span class="fv">{round(t['close_pos_now']*100)}% ({t['cp_bin'].replace('_',' ')})</span></div>
      <div class="feat"><span class="fn">Rango intradiario</span><span class="fv">{t['range_so_far']:.3f}%</span></div>
      <div class="feat"><span class="fn">Retorno primera hora</span><span class="fv" style="color:{color_val(t['first_hour_ret'])}">{t['first_hour_ret']:+.3f}%</span></div>
      <div class="feat"><span class="fn">Retorno mañana (primera mitad)</span><span class="fv" style="color:{color_val(t['morning_ret'])}">{t['morning_ret']:+.3f}%</span></div>
      <div class="feat"><span class="fn">Tendencia tarde vs mañana</span><span class="fv" style="color:{trend_color}">{t['intra_trend']:+.3f}% ({t['intra_bin']})</span></div>
      <div class="feat" style="border:none"><span class="fn">Volumen proyectado</span><span class="fv">{t['vol_rel_proj']:.2f}x promedio</span></div>
      <div style="margin-top:12px;padding:10px;background:#f9fafb;border-radius:8px;font-size:12px;color:#6b7280;line-height:1.6">
        <strong style="color:#374151">Perfil:</strong> día <strong style="color:#374151">{t['ret_bin'].replace('_',' ')} · {t['cp_bin'].replace('_',' ')}</strong> con tendencia intradiaria {t['intra_bin']}. Gap {t['gap_bin'].replace('_',' ')}.
      </div>
    </div>

    <div class="card">
      <div class="sec">Señales condicionales → mañana</div>
      {signals_html}
    </div>
  </div>

  <div>
    <div class="card">
      <div class="sec">Predicción: cierre de hoy</div>
      <div style="font-size:12px;color:#6b7280;margin-bottom:12px">{tp['similar_n']} días con retorno parcial similar a la misma altura de la sesión.</div>
      <div style="display:flex;gap:10px;margin-bottom:12px">
        <div class="mc" style="flex:1;text-align:center"><div class="lb">P(sube desde acá)</div><div class="vl" style="font-size:26px">{round(tp['p_up_from_here']*100,1)}%</div></div>
        <div class="mc" style="flex:1;text-align:center"><div class="lb">Delta esperado</div><div class="vl" style="font-size:26px;color:{delta_color}">{tp['delta_mean']:+.3f}%</div></div>
      </div>
      <div style="padding:10px;background:#f9fafb;border-radius:8px;font-size:12px;color:#6b7280;margin-bottom:10px">
        Cierre proyectado: <strong style="color:#374151">${proj_price:.2f}</strong>
      </div>
      {_verdict_today(tp['p_up_from_here'])}
    </div>

    <div class="card green-border">
      <div class="sec" style="color:#166534">Predicción: mañana ({(datetime.strptime(str(today_date), '%Y-%m-%d').strftime('%d %b %Y') if isinstance(today_date, str) else today_date)}+1)</div>
      <div style="font-size:12px;color:#6b7280;margin-bottom:12px">Top 20 días más similares en 5 dimensiones históricas.</div>
      <div style="display:flex;gap:10px;margin-bottom:12px">
        <div class="mc" style="flex:1;text-align:center"><div class="lb">P(sube mañana)</div><div class="vl" style="font-size:26px;color:{'#166534' if tm['p_up_top20']>0.55 else ('#dc2626' if tm['p_up_top20']<0.45 else '#374151')}">{round(tm['p_up_top20']*100,1)}%</div></div>
        <div class="mc" style="flex:1;text-align:center"><div class="lb">IC 90%</div><div class="vl" style="font-size:18px">[{round(tm['ci_lo']*100)}–{round(tm['ci_hi']*100)}%]</div></div>
      </div>
      <div style="font-size:12px;font-weight:600;margin-bottom:8px;color:#6b7280">Top 5 días más similares</div>
      <table>
        <tr><th>Fecha</th><th style="text-align:center">Ret</th><th style="text-align:center">Gap</th><th style="text-align:center">CP</th><th style="text-align:center">Sig día+1</th><th style="text-align:right">Ret día+1</th></tr>
        {top5_rows}
      </table>
      {_verdict_tomorrow(tm['p_up_top20'], tm['ci_lo'], tm['ci_hi'])}
    </div>
  </div>
</div>

<div class="card">
  <div class="sec">Distribución histórica: retorno parcial → P(sube desde ese punto al cierre)</div>
  <div style="position:relative;width:100%;height:180px"><canvas id="histChart"></canvas></div>
  <div style="font-size:11px;color:#9ca3af;margin-top:8px;line-height:1.6">
    Cada barra = días con retorno parcial (a {t['n_bars_1h']} barras de 1h) en ese rango. Altura = probabilidad de que el precio suba desde ahí hasta el cierre. La barra resaltada en azul es el rango de hoy.
  </div>
</div>

<div class="footer">
  <strong>Metodología:</strong> caracterización del día actual mediante 8 features intradiarias.
  Predicción de cierre: k-NN sobre retorno parcial (±0.3%) a la misma barra del día (n={tp['similar_n']} análogos).
  Predicción mañana: k-NN top-20 por similitud euclidiana normalizada en 5 features (retorno, rango, gap, posición cierre, retorno mañana).
  Señales condicionales: bins univariados con IC Wilson 90% y test-t bilateral.
  Nota: ninguna señal superó el umbral de significancia estadística (p&lt;0.05) en este día típico.
  <br><br>
  <strong>Aviso:</strong> este análisis es puramente estadístico sobre patrones históricos. No es asesoramiento financiero.
  Los patrones pasados no garantizan resultados futuros. Usar como señal complementaria.
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const histLabels  = {hist_labels};
const histPup     = {hist_pup};
const histN       = {hist_n};
const todayBucket = {today_bucket};
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: histLabels,
    datasets: [{{
      data: histPup,
      backgroundColor: histLabels.map((_,i) => i===todayBucket ? '#2563eb' : '#d1d5db'),
      borderRadius: 4, borderSkipped: false,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{display:false}},
      tooltip: {{callbacks: {{
        label: c => {{
          const n=histN[c.dataIndex]; const v=c.parsed.y;
          return v!=null ? ` P(sube): ${{v.toFixed(1)}}% (n=${{n}})` : ' datos insuf.';
        }}
      }}}}
    }},
    scales: {{
      x: {{grid:{{display:false}}, ticks:{{font:{{size:11}},color:'#9ca3af'}}}},
      y: {{min:0,max:100,
           grid:{{color:'rgba(0,0,0,0.05)'}},
           ticks:{{font:{{size:11}},color:'#9ca3af',callback:v=>v+'%'}},
           title:{{display:true,text:'P(sube desde ese punto)',font:{{size:10}},color:'#9ca3af'}}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analiza el día actual del SPY y genera predicciones con reporte HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python spy_live_analysis.py SPY_1h_acum.csv SPY_15m_acum.csv
  python spy_live_analysis.py SPY_1h_acum.csv SPY_15m_acum.csv --output reporte_hoy.html

Los CSV deben tener columnas: datetime, open, high, low, close, volume
y deben incluir los datos parciales del día actual al final.
        """
    )
    parser.add_argument("csv_1h",  help="CSV de datos acumulados con granularidad 1h")
    parser.add_argument("csv_15m", help="CSV de datos acumulados con granularidad 15m")
    parser.add_argument("--output", "-o", default=None,
                        help="Nombre del HTML de salida (default: spy_live_YYYYMMDD.html)")
    args = parser.parse_args()

    for path in [args.csv_1h, args.csv_15m]:
        if not Path(path).exists():
            print(f"ERROR: No se encontró '{path}'")
            sys.exit(1)

    print("Cargando datos...")
    df1h = load_csv(args.csv_1h)
    df15 = load_csv(args.csv_15m)

    today_date = df1h["date"].iloc[-1]
    print(f"  Día actual detectado: {today_date}")
    print(f"  1h: {df1h['date'].nunique()} días · 15m: {df15['date'].nunique()} días")

    print("Construyendo historia diaria...")
    daily = build_daily_history(df1h, today_date)
    base  = float(daily["up"].mean())
    print(f"  {len(daily)} días históricos · base rate: {base:.1%}")

    print("Extrayendo features del día actual...")
    today = extract_today_features(df1h, df15, today_date, daily)
    print(f"  ret={today['ret_so_far']:+.3f}% · cp={today['close_pos_now']:.2f} · gap={today['gap_today']:+.3f}% · trend={today['intra_trend']:+.3f}%")

    print("Predicción cierre de hoy...")
    today_pred = predict_today_close(df1h, today_date, today)
    print(f"  P(sube)={today_pred['p_up_from_here']:.1%} · proj=${today_pred['proj_close']:.2f}")

    print("Predicción para mañana...")
    tmrw = predict_tomorrow(daily, today)
    print(f"  P(sube mañana)={tmrw['p_up_top20']:.1%} · IC=[{tmrw['ci_lo']},{tmrw['ci_hi']}]")

    print("Señal intradiaria...")
    intra = predict_intra_signal(df15, daily, today_date, today)

    print("Generando HTML...")
    html = generate_html(today_date, today, today_pred, tmrw, intra, base, len(daily))

    output = args.output or f"spy_live_{str(today_date).replace('-','')}.html"
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Reporte guardado en: {output}")
    print(f"  Abrilo en cualquier navegador.")
    print(f"\n  Resumen:")
    print(f"  Hoy ({today_date}):   P(cierra más arriba desde acá) = {today_pred['p_up_from_here']:.1%}")
    print(f"  Mañana:              P(sube) = {tmrw['p_up_top20']:.1%}  IC=[{round(tmrw['ci_lo']*100)}%–{round(tmrw['ci_hi']*100)}%]")


if __name__ == "__main__":
    main()
