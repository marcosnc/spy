"""
SPY Max/Min Time Analysis
==========================
Analiza datos intradiarios del SPY (cualquier granularidad: 15m, 30m, 1h)
y genera un reporte HTML autocontenido que muestra:

  1. En qué franjas horarias se forman con mayor frecuencia el máximo y
     el mínimo diario.
  2. Distribución acumulada a lo largo de la jornada.
  3. Tabla detallada por franja con frecuencias absolutas y relativas.
  4. Resumen de estadísticas clave (top horario, segundo pico, etc.)

Dependencias:
    pip install pandas numpy

Uso:
    python spy_analysis_maxmin.py datos_15m.csv
    python spy_analysis_maxmin.py datos_15m.csv --output reporte.html
    python spy_analysis_maxmin.py datos_15m.csv --ticker QQQQQ --tz US/Eastern
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str, tz: str = "America/New_York") -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada. Columnas disponibles: {list(df.columns)}")
    df["date"] = df.index.date
    df["time"] = df.index.strftime("%H:%M")
    return df


def build_daily_extremes(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Para cada día de trading, encuentra la franja horaria donde se registró
    el high absoluto y el low absoluto.
    Devuelve un DataFrame con una fila por día, y la lista de todos los slots.
    """
    all_times = sorted(df["time"].unique())
    records = []

    for date, g in df.groupby("date"):
        g = g.sort_index()
        if len(g) < 4:
            continue

        idx_max = g["high"].idxmax()
        idx_min = g["low"].idxmin()

        records.append({
            "date":      date,
            "max_time":  g.loc[idx_max, "time"],
            "min_time":  g.loc[idx_min, "time"],
            "max_price": g["high"].max(),
            "min_price": g["low"].min(),
            "open":      g["open"].iloc[0],
            "close":     g["close"].iloc[-1],
            "day_ret":   (g["close"].iloc[-1] - g["open"].iloc[0]) / g["open"].iloc[0] * 100,
            "range_pct": (g["high"].max() - g["low"].min()) / g["open"].iloc[0] * 100,
            "n_bars":    len(g),
        })

    daily = pd.DataFrame(records).set_index("date")
    return daily, all_times


def compute_freq_tables(daily: pd.DataFrame, all_times: list) -> dict:
    """Calcula tablas de frecuencia para máximos y mínimos por franja."""
    total = len(daily)

    max_counts = daily["max_time"].value_counts()
    min_counts = daily["min_time"].value_counts()

    rows = []
    for t in all_times:
        mc = int(max_counts.get(t, 0))
        nc = int(min_counts.get(t, 0))
        rows.append({
            "time":       t,
            "max_count":  mc,
            "min_count":  nc,
            "max_pct":    round(mc / total * 100, 1),
            "min_pct":    round(nc / total * 100, 1),
        })

    freq_df = pd.DataFrame(rows).set_index("time")

    # Acumulados
    freq_df["max_cum"] = freq_df["max_count"].cumsum() / total * 100
    freq_df["min_cum"] = freq_df["min_count"].cumsum() / total * 100

    # Sesiones: apertura / mitad / cierre
    n = len(all_times)
    open_slots  = all_times[:max(1, n // 4)]
    close_slots = all_times[max(0, 3 * n // 4):]
    mid_slots   = all_times[n // 4: 3 * n // 4]

    def zone_pct(col, slots):
        return round(freq_df.loc[freq_df.index.isin(slots), col].sum() / total * 100, 1)

    zones = {
        "max": {
            "apertura": zone_pct("max_count", open_slots),
            "mediodia": zone_pct("max_count", mid_slots),
            "cierre":   zone_pct("max_count", close_slots),
        },
        "min": {
            "apertura": zone_pct("min_count", open_slots),
            "mediodia": zone_pct("min_count", mid_slots),
            "cierre":   zone_pct("min_count", close_slots),
        },
    }

    return {
        "freq_df":    freq_df,
        "total_days": total,
        "zones":      zones,
        "all_times":  all_times,
        "max_counts_list": [int(max_counts.get(t, 0)) for t in all_times],
        "min_counts_list": [int(min_counts.get(t, 0)) for t in all_times],
        "max_cum_list":    freq_df["max_cum"].round(1).tolist(),
        "min_cum_list":    freq_df["min_cum"].round(1).tolist(),
    }


def compute_summary(daily: pd.DataFrame, freq: dict) -> dict:
    freq_df = freq["freq_df"]
    total   = freq["total_days"]

    top_max = freq_df["max_count"].idxmax()
    top_min = freq_df["min_count"].idxmax()

    # Segundo pico (excluyendo el primero)
    max2 = freq_df["max_count"].drop(top_max).idxmax()
    min2 = freq_df["min_count"].drop(top_min).idxmax()

    # % de días donde máx y mín son ambos en apertura
    both_open = ((daily["max_time"] == freq["all_times"][0]) &
                 (daily["min_time"] == freq["all_times"][0])).sum()

    # Promedio de franja de máx vs mín (en minutos desde apertura)
    times_list  = freq["all_times"]
    time_to_min = {t: i * _bar_minutes(times_list) for i, t in enumerate(times_list)}
    avg_max_min = daily["max_time"].map(time_to_min).mean()
    avg_min_min = daily["min_time"].map(time_to_min).mean()

    return {
        "top_max_time":  top_max,
        "top_max_pct":   round(freq_df.loc[top_max, "max_count"] / total * 100, 1),
        "top_max_count": int(freq_df.loc[top_max, "max_count"]),
        "top_min_time":  top_min,
        "top_min_pct":   round(freq_df.loc[top_min, "min_count"] / total * 100, 1),
        "top_min_count": int(freq_df.loc[top_min, "min_count"]),
        "second_max_time": max2,
        "second_max_pct":  round(freq_df.loc[max2, "max_count"] / total * 100, 1),
        "second_min_time": min2,
        "second_min_pct":  round(freq_df.loc[min2, "min_count"] / total * 100, 1),
        "both_open_pct": round(both_open / total * 100, 1),
        "avg_max_min":   round(avg_max_min),
        "avg_min_min":   round(avg_min_min),
        "avg_day_ret":   round(daily["day_ret"].mean(), 3),
        "avg_range_pct": round(daily["range_pct"].mean(), 3),
        "open_zone_max": freq["zones"]["max"]["apertura"],
        "open_zone_min": freq["zones"]["min"]["apertura"],
        "close_zone_max": freq["zones"]["max"]["cierre"],
        "close_zone_min": freq["zones"]["min"]["cierre"],
    }


def _bar_minutes(times_list: list) -> int:
    """Infiere la granularidad en minutos a partir de los timestamps."""
    if len(times_list) < 2:
        return 15
    h1, m1 = map(int, times_list[0].split(":"))
    h2, m2 = map(int, times_list[1].split(":"))
    diff = (h2 * 60 + m2) - (h1 * 60 + m1)
    return abs(diff) if abs(diff) > 0 else 15


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERACIÓN DEL HTML
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{ticker} — Análisis de Máximos y Mínimos Intradiarios</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {
    --bg: #f9fafb; --surface: #ffffff; --surface2: #f3f4f6;
    --border: #e5e7eb; --border2: #d1d5db;
    --text: #111827; --muted: #6b7280; --hint: #9ca3af;
    --blue:   #2563eb; --blue-bg:  #dbeafe; --blue-dark: #1e40af;
    --orange: #ea580c; --orange-bg: #ffedd5; --orange-dark: #9a3412;
    --green:  #16a34a; --green-bg: #dcfce7;
    --radius: 10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text);
    font-size: 14px; line-height: 1.5;
  }
  .page { max-width: 1060px; margin: 0 auto; padding: 32px 20px 60px; }

  .header { margin-bottom: 28px; }
  .header h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .header p  { color: var(--muted); font-size: 13px; }

  /* metric cards */
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
  .mc { background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 14px 16px; }
  .mc.blue-accent   { border-left: 4px solid var(--blue); }
  .mc.orange-accent { border-left: 4px solid var(--orange); }
  .mc .lbl { font-size: 11px; color: var(--muted); text-transform: uppercase;
             letter-spacing: .05em; margin-bottom: 4px; }
  .mc .val { font-size: 26px; font-weight: 700; }
  .mc .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* section title */
  .sec { font-size: 11px; font-weight: 700; color: var(--muted);
         text-transform: uppercase; letter-spacing: .07em; margin-bottom: 14px; }

  /* chart cards */
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 20px 24px; margin-bottom: 20px; }
  .chart-wrap        { position: relative; width: 100%; height: 280px; }
  .chart-wrap-tall   { position: relative; width: 100%; height: 200px; }

  /* legend row */
  .legend { display: flex; gap: 20px; margin-bottom: 14px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px;
                 font-size: 12px; color: var(--muted); }
  .legend-swatch { width: 12px; height: 12px; border-radius: 2px; }

  /* two-col grid */
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }

  /* zone bars */
  .zones { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
  .zone-block h4 { font-size: 12px; font-weight: 600; margin-bottom: 8px; color: var(--muted); }
  .zone-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
  .zone-name { min-width: 70px; color: var(--muted); }
  .zone-bar-bg { flex: 1; background: var(--border); border-radius: 3px; height: 6px; overflow: hidden; }
  .zone-bar-fill { height: 100%; border-radius: 3px; }
  .zone-val { min-width: 34px; text-align: right; font-weight: 600; font-size: 12px; }

  /* frequency table */
  .freq-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .freq-table th { text-align: left; color: var(--muted); font-weight: 600;
                   font-size: 11px; padding: 7px 10px;
                   border-bottom: 2px solid var(--border); background: var(--surface2); }
  .freq-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); }
  .freq-table tr:last-child td { border-bottom: none; }
  .freq-table tr:hover { background: var(--surface2); }
  .freq-table .bar-cell { width: 120px; }
  .mini-bar-bg { background: var(--border); border-radius: 2px; height: 5px;
                 width: 100px; overflow: hidden; margin-top: 3px; }
  .mini-bar    { height: 100%; border-radius: 2px; }
  .rank-badge  { display: inline-block; font-size: 10px; font-weight: 700;
                 padding: 1px 6px; border-radius: 3px; }
  .rank-1 { background: #fef08a; color: #713f12; }
  .rank-2 { background: #e2e8f0; color: #475569; }
  .rank-3 { background: #fed7aa; color: #9a3412; }

  /* insight boxes */
  .insights { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 20px; }
  .insight { background: var(--surface); border: 1px solid var(--border);
             border-radius: var(--radius); padding: 14px 16px; }
  .insight.blue   { border-top: 3px solid var(--blue); }
  .insight.orange { border-top: 3px solid var(--orange); }
  .insight h3 { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
  .insight p  { font-size: 12px; color: var(--muted); line-height: 1.6; }
  .insight .big { font-size: 22px; font-weight: 700; margin: 4px 0; }

  .footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
            font-size: 11px; color: var(--hint); line-height: 1.7; }

  @media (max-width: 700px) {
    .metrics { grid-template-columns: 1fr 1fr; }
    .grid2   { grid-template-columns: 1fr; }
    .insights{ grid-template-columns: 1fr; }
    .zones   { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <div class="header">
    <h1>{ticker} — Análisis de Máximos y Mínimos Intradiarios</h1>
    <p>Generado el {generated_at} &nbsp;·&nbsp; Datos: {date_range} &nbsp;·&nbsp;
       {total_days} días de trading &nbsp;·&nbsp; Granularidad: {granularity} min &nbsp;·&nbsp; Horario: ET</p>
  </div>

  <!-- METRIC CARDS -->
  <div class="metrics">
    <div class="mc blue-accent">
      <div class="lbl">Top horario — Máximos</div>
      <div class="val" style="color:var(--blue);">{top_max_time}</div>
      <div class="sub">{top_max_count} días · {top_max_pct}% del total</div>
    </div>
    <div class="mc blue-accent">
      <div class="lbl">2° horario — Máximos</div>
      <div class="val" style="color:var(--blue); font-size:20px;">{second_max_time}</div>
      <div class="sub">{second_max_pct}% del total</div>
    </div>
    <div class="mc orange-accent">
      <div class="lbl">Top horario — Mínimos</div>
      <div class="val" style="color:var(--orange);">{top_min_time}</div>
      <div class="sub">{top_min_count} días · {top_min_pct}% del total</div>
    </div>
    <div class="mc orange-accent">
      <div class="lbl">2° horario — Mínimos</div>
      <div class="val" style="color:var(--orange); font-size:20px;">{second_min_time}</div>
      <div class="sub">{second_min_pct}% del total</div>
    </div>
  </div>

  <!-- CHART PRINCIPAL: frecuencias side-by-side -->
  <div class="card">
    <div class="sec">Frecuencia de máximos y mínimos diarios por franja horaria</div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-swatch" style="background:#2563eb;"></div>
        Máximos diarios
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#ea580c;"></div>
        Mínimos diarios
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="mainChart"></canvas>
    </div>
  </div>

  <!-- GRÁFICO SOLO MÁXIMOS + SOLO MÍNIMOS -->
  <div class="grid2">
    <div class="card">
      <div class="sec" style="color:var(--blue-dark);">Distribución de máximos</div>
      <div class="chart-wrap-tall">
        <canvas id="maxChart"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="sec" style="color:var(--orange-dark);">Distribución de mínimos</div>
      <div class="chart-wrap-tall">
        <canvas id="minChart"></canvas>
      </div>
    </div>
  </div>

  <!-- CURVAS ACUMULADAS -->
  <div class="card">
    <div class="sec">Distribución acumulada — ¿a qué hora ya se formó el extremo?</div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-swatch" style="background:#2563eb;"></div>
        % días cuyo máximo ya ocurrió
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#ea580c;"></div>
        % días cuyo mínimo ya ocurrió
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="cumChart"></canvas>
    </div>
  </div>

  <!-- INSIGHTS -->
  <div class="insights">
    <div class="insight blue">
      <h3>Zona de apertura (primer cuarto de la sesión)</h3>
      <div class="big" style="color:var(--blue);">{open_zone_max}%</div>
      <p>de los máximos diarios se forman en la apertura. {open_zone_min}% de los mínimos también. La primera hora concentra la mayor volatilidad.</p>
    </div>
    <div class="insight orange">
      <h3>Zona de cierre (último cuarto de la sesión)</h3>
      <div class="big" style="color:var(--orange);">{close_zone_max}%</div>
      <p>de los máximos se forman al cierre. {close_zone_min}% de los mínimos. El rebalanceo institucional genera el segundo pico de actividad.</p>
    </div>
    <div class="insight blue">
      <h3>Tiempo promedio al máximo</h3>
      <div class="big" style="color:var(--blue);">{avg_max_min} min</div>
      <p>desde la apertura hasta que se registra el high del día (promedio). Equivale a la {avg_max_bar}ª barra de la sesión.</p>
    </div>
    <div class="insight orange">
      <h3>Tiempo promedio al mínimo</h3>
      <div class="big" style="color:var(--orange);">{avg_min_min} min</div>
      <p>desde la apertura hasta que se registra el low del día (promedio). Equivale a la {avg_min_bar}ª barra de la sesión.</p>
    </div>
  </div>

  <!-- DISTRIBUCIÓN POR ZONAS -->
  <div class="card">
    <div class="sec">Concentración por zona de la sesión</div>
    <div class="zones">
      <div class="zone-block">
        <h4>Máximos por zona</h4>
        {max_zone_bars}
      </div>
      <div class="zone-block">
        <h4>Mínimos por zona</h4>
        {min_zone_bars}
      </div>
    </div>
  </div>

  <!-- TABLA DE FRECUENCIAS -->
  <div class="card">
    <div class="sec">Tabla completa por franja horaria</div>
    <table class="freq-table">
      <thead>
        <tr>
          <th>Franja</th>
          <th>Máximos (n)</th>
          <th>Máximos (%)</th>
          <th style="width:120px;">Distribución máx.</th>
          <th>Mínimos (n)</th>
          <th>Mínimos (%)</th>
          <th style="width:120px;">Distribución mín.</th>
          <th>Acum. máx.</th>
          <th>Acum. mín.</th>
        </tr>
      </thead>
      <tbody>
        {freq_table_rows}
      </tbody>
    </table>
  </div>

  <div class="footer">
    <strong>Metodología:</strong> para cada día de trading se identifica la barra (vela) donde se registra
    el high absoluto del día (usando la columna <em>high</em>) y la barra donde se registra el low absoluto
    (usando la columna <em>low</em>). Se contabiliza la franja horaria de inicio de cada barra.
    En caso de empate se toma la primera ocurrencia. Días con menos de 4 barras son excluidos.
    <br><br>
    <strong>Aviso:</strong> este análisis es descriptivo de comportamiento histórico y no constituye
    asesoramiento financiero. Los patrones pasados no garantizan resultados futuros.
  </div>
</div>

<script>
const times     = {times_json};
const maxCounts = {max_counts_json};
const minCounts = {min_counts_json};
const maxCum    = {max_cum_json};
const minCum    = {min_cum_json};
const total     = {total_days};

// Colores dinámicos según intensidad
function blueAlpha(v) {
  const maxV = Math.max(...maxCounts);
  const a = 0.25 + 0.75 * (v / maxV);
  return `rgba(37, 99, 235, ${a.toFixed(2)})`;
}
function orangeAlpha(v) {
  const maxV = Math.max(...minCounts);
  const a = 0.25 + 0.75 * (v / maxV);
  return `rgba(234, 88, 12, ${a.toFixed(2)})`;
}

// ─── Gráfico principal (side-by-side) ────────────────────────────────────────
new Chart(document.getElementById('mainChart'), {
  type: 'bar',
  data: {
    labels: times,
    datasets: [
      {
        label: 'Máximos',
        data: maxCounts,
        backgroundColor: '#2563eb',
        borderRadius: 3,
        borderSkipped: false,
      },
      {
        label: 'Mínimos',
        data: minCounts,
        backgroundColor: '#ea580c',
        borderRadius: 3,
        borderSkipped: false,
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: c => ` ${c.dataset.label}: ${c.parsed.y} días (${(c.parsed.y/total*100).toFixed(1)}%)`
        }
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          autoSkip: false, maxRotation: 45,
          font: { size: 10 }, color: '#9ca3af',
          callback: (_, i) => {
            const t = times[i];
            const [h, m] = t.split(':').map(Number);
            return m === 0 || t === times[0] || t === times[times.length-1] ? t : '';
          }
        }
      },
      y: {
        beginAtZero: true,
        grid: { color: 'rgba(0,0,0,0.06)' },
        ticks: { font: { size: 11 }, color: '#9ca3af' },
        title: { display: true, text: 'Cantidad de días', font: { size: 11 }, color: '#9ca3af' }
      }
    }
  }
});

// ─── Solo máximos ────────────────────────────────────────────────────────────
new Chart(document.getElementById('maxChart'), {
  type: 'bar',
  data: {
    labels: times,
    datasets: [{
      data: maxCounts,
      backgroundColor: maxCounts.map(v => blueAlpha(v)),
      borderRadius: 3, borderSkipped: false,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ` ${c.parsed.y} días (${(c.parsed.y/total*100).toFixed(1)}%)` } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 45, font: { size: 9 }, color: '#9ca3af',
             callback: (_, i) => { const [h,m] = times[i].split(':').map(Number); return m===0||i===0||i===times.length-1?times[i]:''; } } },
      y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 10 }, color: '#9ca3af' } }
    }
  }
});

// ─── Solo mínimos ────────────────────────────────────────────────────────────
new Chart(document.getElementById('minChart'), {
  type: 'bar',
  data: {
    labels: times,
    datasets: [{
      data: minCounts,
      backgroundColor: minCounts.map(v => orangeAlpha(v)),
      borderRadius: 3, borderSkipped: false,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ` ${c.parsed.y} días (${(c.parsed.y/total*100).toFixed(1)}%)` } }
    },
    scales: {
      x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 45, font: { size: 9 }, color: '#9ca3af',
             callback: (_, i) => { const [h,m] = times[i].split(':').map(Number); return m===0||i===0||i===times.length-1?times[i]:''; } } },
      y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' }, ticks: { font: { size: 10 }, color: '#9ca3af' } }
    }
  }
});

// ─── Curvas acumuladas ───────────────────────────────────────────────────────
new Chart(document.getElementById('cumChart'), {
  type: 'line',
  data: {
    labels: times,
    datasets: [
      {
        label: 'Máximos acumulados',
        data: maxCum,
        borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3,
      },
      {
        label: 'Mínimos acumulados',
        data: minCum,
        borderColor: '#ea580c', backgroundColor: 'rgba(234,88,12,0.08)',
        borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3,
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ` ${c.dataset.label}: ${c.parsed.y.toFixed(1)}% de los días` } }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          autoSkip: false, maxRotation: 45, font: { size: 10 }, color: '#9ca3af',
          callback: (_, i) => { const [h,m] = times[i].split(':').map(Number); return m===0||i===0||i===times.length-1?times[i]:''; }
        }
      },
      y: {
        min: 0, max: 100,
        grid: { color: 'rgba(0,0,0,0.06)' },
        ticks: { font: { size: 11 }, color: '#9ca3af', callback: v => v + '%' },
        title: { display: true, text: '% días con extremo ya formado', font: { size: 11 }, color: '#9ca3af' }
      }
    }
  }
});
</script>
</body>
</html>
"""


def build_zone_bars_html(zones_dict, color):
    zone_names = {"apertura": "Apertura", "mediodia": "Mediodía", "cierre": "Cierre"}
    html = ""
    for key, label in zone_names.items():
        pct = zones_dict[key]
        fill_color = "#2563eb" if color == "blue" else "#ea580c"
        html += f"""
        <div class="zone-row">
          <span class="zone-name">{label}</span>
          <div class="zone-bar-bg">
            <div class="zone-bar-fill" style="width:{min(pct,100)}%;background:{fill_color};"></div>
          </div>
          <span class="zone-val" style="color:{fill_color};">{pct}%</span>
        </div>"""
    return html


def build_freq_table_rows(freq_df, total):
    # ranks
    max_rank = {t: r + 1 for r, t in enumerate(freq_df["max_count"].sort_values(ascending=False).index)}
    min_rank = {t: r + 1 for r, t in enumerate(freq_df["min_count"].sort_values(ascending=False).index)}
    max_max  = freq_df["max_count"].max()
    min_max  = freq_df["min_count"].max()

    def badge(rank):
        if rank == 1: return '<span class="rank-badge rank-1">1°</span>'
        if rank == 2: return '<span class="rank-badge rank-2">2°</span>'
        if rank == 3: return '<span class="rank-badge rank-3">3°</span>'
        return ""

    rows = ""
    for t, row in freq_df.iterrows():
        mc = int(row["max_count"])
        nc = int(row["min_count"])
        mr = max_rank[t]
        nr = min_rank[t]
        mw = round(mc / max_max * 100) if max_max > 0 else 0
        nw = round(nc / min_max * 100) if min_max > 0 else 0
        rows += f"""
        <tr>
          <td><strong>{t}</strong></td>
          <td>{mc} {badge(mr)}</td>
          <td>{round(mc/total*100,1)}%</td>
          <td class="bar-cell">
            <div class="mini-bar-bg">
              <div class="mini-bar" style="width:{mw}%;background:#2563eb;"></div>
            </div>
          </td>
          <td>{nc} {badge(nr)}</td>
          <td>{round(nc/total*100,1)}%</td>
          <td class="bar-cell">
            <div class="mini-bar-bg">
              <div class="mini-bar" style="width:{nw}%;background:#ea580c;"></div>
            </div>
          </td>
          <td>{round(row['max_cum'],1)}%</td>
          <td>{round(row['min_cum'],1)}%</td>
        </tr>"""
    return rows


def generate_html(df, daily, freq, summary, ticker, csv_path):
    gran = _bar_minutes(freq["all_times"])
    date_range = f"{daily.index[0]} → {daily.index[-1]}"

    replacements = {
        "{ticker}":          ticker,
        "{generated_at}":    datetime.now().strftime("%d/%m/%Y %H:%M"),
        "{date_range}":      date_range,
        "{total_days}":      str(freq["total_days"]),
        "{granularity}":     str(gran),
        "{top_max_time}":    summary["top_max_time"],
        "{top_max_count}":   str(summary["top_max_count"]),
        "{top_max_pct}":     str(summary["top_max_pct"]),
        "{second_max_time}": summary["second_max_time"],
        "{second_max_pct}":  str(summary["second_max_pct"]),
        "{top_min_time}":    summary["top_min_time"],
        "{top_min_count}":   str(summary["top_min_count"]),
        "{top_min_pct}":     str(summary["top_min_pct"]),
        "{second_min_time}": summary["second_min_time"],
        "{second_min_pct}":  str(summary["second_min_pct"]),
        "{open_zone_max}":   str(summary["open_zone_max"]),
        "{open_zone_min}":   str(summary["open_zone_min"]),
        "{close_zone_max}":  str(summary["close_zone_max"]),
        "{close_zone_min}":  str(summary["close_zone_min"]),
        "{avg_max_min}":     str(summary["avg_max_min"]),
        "{avg_min_min}":     str(summary["avg_min_min"]),
        "{avg_max_bar}":     str(round(summary["avg_max_min"] / gran) + 1),
        "{avg_min_bar}":     str(round(summary["avg_min_min"] / gran) + 1),
        "{max_zone_bars}":   build_zone_bars_html(freq["zones"]["max"], "blue"),
        "{min_zone_bars}":   build_zone_bars_html(freq["zones"]["min"], "orange"),
        "{freq_table_rows}": build_freq_table_rows(freq["freq_df"], freq["total_days"]),
        "{times_json}":      json.dumps(freq["all_times"]),
        "{max_counts_json}": json.dumps(freq["max_counts_list"]),
        "{min_counts_json}": json.dumps(freq["min_counts_list"]),
        "{max_cum_json}":    json.dumps(freq["max_cum_list"]),
        "{min_cum_json}":    json.dumps(freq["min_cum_list"]),
    }

    html = HTML_TEMPLATE
    for key, val in replacements.items():
        html = html.replace(key, val)
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analiza datos intradiarios del SPY y genera un reporte HTML de máximos/mínimos por hora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python spy_analysis_maxmin.py SPY_15m.csv
  python spy_analysis_maxmin.py SPY_15m.csv --output reporte_maxmin.html
  python spy_analysis_maxmin.py SPY_30m.csv --ticker QQQ
  python spy_analysis_maxmin.py SPY_15m.csv --tz US/Eastern
        """
    )
    parser.add_argument("csv",            help="Ruta al archivo CSV de datos intradiarios")
    parser.add_argument("--output", "-o", default=None,
                        help="Nombre del HTML de salida (default: spy_maxmin_YYYYMMDD_HHMMSS.html)")
    parser.add_argument("--ticker",       default="SPY",
                        help="Nombre del ticker para el reporte (default: SPY)")
    parser.add_argument("--tz",           default="America/New_York",
                        help="Timezone de los datos (default: America/New_York)")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: No se encontró '{args.csv}'")
        sys.exit(1)

    output_path = args.output or f"spy_maxmin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    print(f"Cargando datos: {args.csv}")
    df = load_data(args.csv, tz=args.tz)
    print(f"  → {len(df)} barras · {df['date'].nunique()} días en el CSV")

    print("Calculando extremos diarios...")
    daily, all_times = build_daily_extremes(df)
    print(f"  → {len(daily)} días de trading válidos · {len(all_times)} franjas horarias")

    print("Calculando frecuencias...")
    freq = compute_freq_tables(daily, all_times)

    print("Calculando estadísticas resumen...")
    summary = compute_summary(daily, freq)
    print(f"  → Top máx: {summary['top_max_time']} ({summary['top_max_pct']}%) · "
          f"Top mín: {summary['top_min_time']} ({summary['top_min_pct']}%)")

    print("Generando HTML...")
    html = generate_html(df, daily, freq, summary, args.ticker, args.csv)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✓ Reporte guardado en: {output_path}")
    print(f"  Abrilo con cualquier navegador (Chrome, Firefox, Safari, Edge).")


if __name__ == "__main__":
    main()
