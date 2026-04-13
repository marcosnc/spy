"""
SPY Opening Analysis — Predictor de Umbrales desde un Precio de Referencia
===========================================================================
Calcula la probabilidad de que el SPY cierre +N% o mas arriba / -N% o mas
abajo de un precio de referencia, usando contexto historico (k-NN + señales
condicionales).

Por defecto el precio de referencia es el precio de apertura del dia actual.
Con --precio se puede especificar cualquier otro precio (por ejemplo el precio
actual a mitad de sesion, un nivel tecnico, etc.).

Las probabilidades se calculan siempre sobre el CIERRE del dia vs el precio
de referencia dado. El historial se recalibra en funcion de ese precio.

Uso:
    python spy_open_analysis.py datos_1h_acum.csv
    python spy_open_analysis.py datos_1h_acum.csv --precio 660.00
    python spy_open_analysis.py datos_1h_acum.csv --precio 650.50 --umbral 0.75
    python spy_open_analysis.py datos_1h_acum.csv --output reporte.html

Dependencias:
    pip install pandas numpy scipy
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm


# ── 1. CARGA Y FEATURES ──────────────────────────────────────────────────────

def load_and_build(csv_path, umbral=0.5, precio_ref=None):
    """
    precio_ref: si es None usa el open del dia actual como referencia.
    Si es un float, recalibra los targets y el historial vs ese precio.

    Logica de recalibracion:
      El historial registra retornos desde el open de cada dia historico.
      Queremos saber P(close >= ref*(1+um/100)).
      En terminos del retorno historico desde el open del dia historico:
        P(ret_hist >= (ref*(1+um/100) / open_hist - 1)*100)
      Esto ajusta el umbral efectivo segun la distancia entre ref y el open
      de cada dia historico, lo que es correcto y consistente.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = df.index.date
    today_date = df["date"].iloc[-1]

    rows = []
    for date, g in df.groupby("date"):
        g = g.sort_index()
        if len(g) < 1: continue
        o=float(g["open"].iloc[0]); c=float(g["close"].iloc[-1])
        h=float(g["high"].max());   l=float(g["low"].min())
        rows.append({"date":date,"open":o,"close":c,"high":h,"low":l,
            "vol":float(g["volume"].sum()),"n_bars":len(g),"is_today":(date==today_date),
            "ret":(c-o)/o*100,"range":(h-l)/o*100,
            "ih":(h-o)/o*100,"il":(l-o)/o*100,
        })

    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    n = len(d)
    gap=[np.nan]*n; prev=[np.nan]*n; vol5=[np.nan]*n; ret3d=[np.nan]*n; vrel=[np.nan]*n

    for i in range(1, n):
        pc=d.loc[i-1,"close"]; co=d.loc[i,"open"]
        gap[i]=(co-pc)/pc*100
        prev[i]=d.loc[i-1,"ret"]
        vm20=d.loc[max(0,i-20):i-1,"vol"].mean()
        vrel[i]=d.loc[i,"vol"]/vm20 if vm20>0 else 1.0
        r3=d.loc[max(0,i-3):i-1,"ret"]
        ret3d[i]=float(r3.sum()) if len(r3)>0 else np.nan
        r5=d.loc[max(0,i-5):i-1,"ret"]
        vol5[i]=float(r5.std()) if len(r5)>=3 else np.nan

    d["gap"]=gap; d["prev"]=prev; d["vol5"]=vol5; d["ret3d"]=ret3d; d["vrel"]=vrel

    today_open = float(d[d["is_today"]]["open"].iloc[0])
    ref = precio_ref if precio_ref is not None else today_open

    # Recalibrar cp/cm del historial usando precio de referencia.
    # Umbral efectivo sobre la distribucion de retornos desde el open:
    #   ret_threshold_up = (ref/today_open - 1)*100 + umbral
    #   ret_threshold_dn = (ref/today_open - 1)*100 - umbral
    # Cuando ref == today_open: coincide con el caso base (+-umbral%).
    # Cuando ref > today_open: el umbral alcista es mayor (mas dificil de alcanzar).
    # Cuando ref < today_open: el umbral alcista es menor (mas facil de alcanzar).
    ref_shift = (ref / today_open - 1) * 100   # desplazamiento del precio de referencia vs open
    thr_up = ref_shift + umbral                 # umbral efectivo alcista (% desde open)
    thr_dn = ref_shift - umbral                 # umbral efectivo bajista (% desde open)

    hist_rows = d[~d["is_today"]].copy().reset_index(drop=True)
    hist_rows["cp"] = (hist_rows["ret"] >= thr_up).astype(int)
    hist_rows["cm"] = (hist_rows["ret"] <= thr_dn).astype(int)

    hist = hist_rows.dropna(subset=["gap","prev","vol5","ret3d"]).reset_index(drop=True)
    tr = d[d["is_today"]].iloc[0]

    today = {
        "date":         str(today_date),
        "open":         round(today_open, 2),
        "precio_ref":   round(ref, 2),
        "precio_custom":precio_ref is not None,
        "gap":          round(float(tr["gap"]), 4),
        "prev":         round(float(tr["prev"]), 4),
        "vol5":         round(float(tr["vol5"]), 5),
        "ret3d":        round(float(tr["ret3d"]), 4),
        "target_up":    round(ref * (1 + umbral/100), 2),
        "target_dn":    round(ref * (1 - umbral/100), 2),
        "umbral":       umbral,
    }
    return hist, today


# ── 2. ANÁLISIS ───────────────────────────────────────────────────────────────

def _wci(p, n, conf=0.90):
    if n==0: return 0.0,1.0
    z=norm.ppf((1+conf)/2); d=1+z**2/n
    c=(p+z**2/(2*n))/d; m=z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return round(max(0,c-m),3),round(min(1,c+m),3)

def run_analysis(hist, today):
    um=today["umbral"]
    base_pp=float(hist["cp"].mean()); base_pm=float(hist["cm"].mean())

    # KNN
    fc=["gap","prev","vol5","ret3d"]
    mat=hist[fc].values.astype(float)
    tv=np.array([today["gap"],today["prev"],today["vol5"],today["ret3d"]])
    st=mat.std(axis=0); st[st==0]=1
    ds=np.sqrt(((mat-tv)/st)**2).sum(axis=1)
    idx30=np.argsort(ds)[:30]
    t30=hist.iloc[idx30].copy(); t30["dist"]=ds[idx30]

    kpp=float(t30["cp"].mean()); kpm=float(t30["cm"].mean())
    kmr=float(t30["ret"].mean()); ksr=float(t30["ret"].std())
    ci_pp=_wci(kpp,30); ci_pm=_wci(kpm,30)

    # Señales condicionales
    def cond(col,lo,hi):
        g=hist[(hist[col]>=lo)&(hist[col]<hi)]
        if len(g)<6: return None
        pp=float(g["cp"].mean()); pm=float(g["cm"].mean())
        return {"n":len(g),"pp":round(pp,4),"pm":round(pm,4),
                "mr":round(float(g["ret"].mean()),4),"sr":round(float(g["ret"].std()),4),
                "ci_pp":_wci(pp,len(g)),"ci_pm":_wci(pm,len(g))}

    cg={}
    for lo,hi,k in [(-99,-.5,"fbajo"),(-.5,-.15,"lbajo"),(-.15,.15,"neutro"),(.15,.5,"lalto"),(.5,99,"falto")]:
        r=cond("gap",lo,hi);
        if r: cg[k]=r
    cp2={}
    for lo,hi,k in [(-99,-1,"mybajo"),(-1,-.3,"bajo"),(-.3,.3,"plano"),(.3,1,"alto"),(1,99,"myalto")]:
        r=cond("prev",lo,hi)
        if r: cp2[k]=r
    vm=float(np.nanmedian(hist["vol5"]))
    cv={}
    for lo,hi,k in [(0,vm*.7,"baja"),(vm*.7,vm*1.3,"media"),(vm*1.3,99,"alta")]:
        r=cond("vol5",lo,hi)
        if r: cv[k]=r

    tg,tp,tv5=today["gap"],today["prev"],today["vol5"]
    gb=("fbajo" if tg<-.5 else "lbajo" if tg<-.15 else "neutro" if tg<.15 else "lalto" if tg<.5 else "falto")
    pb=("mybajo" if tp<-1 else "bajo" if tp<-.3 else "plano" if tp<.3 else "alto" if tp<1 else "myalto")
    vb=("baja" if tv5<vm*.7 else "media" if tv5<vm*1.3 else "alta")

    bins=np.arange(-3.0,3.26,0.25)
    def mhist(s):
        return [{"lo":round(float(lo),2),"n":int(((s>=lo)&(s<lo+0.25)).sum())} for lo in bins[:-1]]

    pcts=[5,10,25,50,75,90,95]
    ap=[round(float(np.percentile(hist["ret"],p)),3) for p in pcts]
    tp30=[round(float(np.percentile(t30["ret"],p)),3) for p in pcts]

    t30l=[{"date":str(r["date"]),"gap":round(r["gap"],3),"prev":round(r["prev"],3),
           "vol5":round(r["vol5"],4),"ret":round(r["ret"],3),
           "p05":int(r["cp"]),"m05":int(r["cm"]),"dist":round(r["dist"],3)}
          for _,r in t30.iterrows()]

    return {
        "base":{"pp":round(base_pp,4),"pm":round(base_pm,4),
                "mr":round(float(hist["ret"].mean()),4),"sr":round(float(hist["ret"].std()),4),
                "n":len(hist),"ci_pp":_wci(base_pp,len(hist)),"ci_pm":_wci(base_pm,len(hist))},
        "knn":{"pp":round(kpp,4),"pm":round(kpm,4),"mr":round(kmr,4),"sr":round(ksr,4),
               "ci_pp":ci_pp,"ci_pm":ci_pm,"days":t30l},
        "cond_gap":cg,"cond_prev":cp2,"cond_vol":cv,
        "gap_bin":gb,"prev_bin":pb,"vol_bin":vb,"vol_med":round(vm,5),
        "hist_bins":[round(float(b),2) for b in bins[:-1]],
        "hist_all":mhist(hist["ret"]),"hist_top30":mhist(t30["ret"]),
        "pcts":{"vals":pcts,"all":ap,"top30":tp30},
    }


# ── 3. HTML ───────────────────────────────────────────────────────────────────

def _pbar(p, base, ci, color="#16a34a"):
    pct=round(p*100,1)
    bc=color if abs(p-base)>0.03 else "#94a3b8"
    return (f'<div style="display:flex;align-items:center;gap:8px">'
            f'<span style="font-size:14px;font-weight:700;min-width:48px;color:{bc}">{pct}%</span>'
            f'<div style="flex:1;background:#e5e7eb;border-radius:4px;height:10px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;border-radius:4px;background:{bc}"></div></div>'
            f'<span style="font-size:11px;color:#9ca3af">[{round(ci[0]*100)}–{round(ci[1]*100)}%]</span>'
            f'</div>')

def _crow(label, sig, base, today_bin=False):
    if not sig: return ""
    pp=sig["pp"]; pm=sig["pm"]; n=sig["n"]
    hl="background:#eff6ff;" if today_bin else ""
    mark="→ " if today_bin else ""
    ppc=("#16a34a" if pp>base+0.03 else "#dc2626" if pp<base-0.03 else "#374151")
    pmc=("#dc2626" if pm>base+0.03 else "#16a34a" if pm<base-0.03 else "#374151")
    return (f'<tr style="{hl}">'
            f'<td style="padding:6px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;color:#374151">{mark}{label}</td>'
            f'<td style="padding:6px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;font-weight:700;color:{ppc};text-align:center">{round(pp*100,1)}%</td>'
            f'<td style="padding:6px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;font-weight:700;color:{pmc};text-align:center">{round(pm*100,1)}%</td>'
            f'<td style="padding:6px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;color:#9ca3af;text-align:center">{n}</td>'
            f'</tr>')

def _verdict_box(label, p, base, ci, up=True):
    pct=round(p*100,1); diff=round((p-base)*100,1)
    if up:
        bg,border,color=("#dcfce7","#86efac","#166534") if p>=base+0.04 else ("#fee2e2","#fca5a5","#991b1b") if p<=base-0.04 else ("#f9fafb","#e5e7eb","#374151")
    else:
        bg,border,color=("#fee2e2","#fca5a5","#991b1b") if p>=base+0.04 else ("#dcfce7","#86efac","#166534") if p<=base-0.04 else ("#f9fafb","#e5e7eb","#374151")
    arrow="▲" if p>base+0.04 else ("▼" if p<base-0.04 else "~")
    return (f'<div style="padding:16px 20px;border-radius:12px;background:{bg};border:1.5px solid {border};margin-bottom:12px">'
            f'<div style="font-size:11px;color:{color};font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">{label}</div>'
            f'<div style="font-size:40px;font-weight:800;color:{color};line-height:1">{pct}%</div>'
            f'<div style="font-size:12px;color:{color};margin-top:6px">{arrow} {diff:+.1f} pp vs base rate ({round(base*100,1)}%)</div>'
            f'<div style="font-size:11px;color:{color};margin-top:2px">IC 90%: [{round(ci[0]*100)}% – {round(ci[1]*100)}%]</div>'
            f'</div>')

def generate_html(today, res):
    um=today["umbral"]
    bpp=res["base"]["pp"]; bpm=res["base"]["pm"]
    kpp=res["knn"]["pp"];   kpm=res["knn"]["pm"]
    ci_pp=res["knn"]["ci_pp"]; ci_pm=res["knn"]["ci_pm"]

    GL={"fbajo":"Gap bajista fuerte (<-0.5%)","lbajo":"Gap bajista leve (-0.5% a -0.15%)",
        "neutro":"Gap neutro (±0.15%)","lalto":"Gap alcista leve (+0.15% a +0.5%)","falto":"Gap alcista fuerte (>+0.5%)"}
    PL={"mybajo":"Día anterior muy bajista (<-1%)","bajo":"Día anterior bajista (-1% a -0.3%)",
        "plano":"Día anterior plano (±0.3%)","alto":"Día anterior alcista (+0.3% a +1%)","myalto":"Día anterior muy alcista (>+1%)"}
    VL={"baja":"Volatilidad reciente baja","media":"Volatilidad reciente media","alta":"Volatilidad reciente alta"}

    crow_gap="".join(_crow(GL[k],res["cond_gap"].get(k),bpp,k==res["gap_bin"]) for k in GL if k in res["cond_gap"])
    crow_prev="".join(_crow(PL[k],res["cond_prev"].get(k),bpp,k==res["prev_bin"]) for k in PL if k in res["cond_prev"])
    crow_vol="".join(_crow(VL[k],res["cond_vol"].get(k),bpp,k==res["vol_bin"]) for k in VL if k in res["cond_vol"])

    top5=""
    for d in res["knn"]["days"][:5]:
        rc="#16a34a" if d["ret"]>0 else "#dc2626"
        top5+=(f'<tr><td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px">{d["date"]}</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:center">{d["gap"]:+.2f}%</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:center">{d["prev"]:+.2f}%</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:center">{d["vol5"]:.3f}</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:right;color:{rc};font-weight:700">{d["ret"]:+.2f}%</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:center;color:{"#16a34a" if d["p05"] else "#9ca3af"}">{"Si" if d["p05"] else "No"}</td>'
               f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:center;color:{"#dc2626" if d["m05"] else "#9ca3af"}">{"Si" if d["m05"] else "No"}</td></tr>')

    pct_rows=""
    for p,a,t in zip(res["pcts"]["vals"],res["pcts"]["all"],res["pcts"]["top30"]):
        pct_rows+=(f'<tr><td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;color:#6b7280">{p}%</td>'
                   f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:right;font-weight:600;color:{"#16a34a" if a>0 else "#dc2626"}">{a:+.3f}%</td>'
                   f'<td style="padding:5px 8px;border-bottom:0.5px solid #f3f4f6;font-size:12px;text-align:right;font-weight:600;color:{"#16a34a" if t>0 else "#dc2626"}">{t:+.3f}%</td></tr>')

    hbins=json.dumps(res["hist_bins"])
    hall=json.dumps([h["n"] for h in res["hist_all"]])
    ht30=json.dumps([h["n"] for h in res["hist_top30"]])

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>SPY Apertura — {today["date"]}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f9fafb;color:#111827;font-size:14px;line-height:1.5}}
  .page{{max-width:1100px;margin:0 auto;padding:32px 20px 60px}}
  h1{{font-size:22px;font-weight:700;margin-bottom:4px}}
  .sub{{color:#6b7280;font-size:13px;margin-bottom:24px}}
  .g4{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:20px}}
  .g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}}
  .mc{{background:#f3f4f6;border-radius:8px;padding:12px 14px}}
  .mc .lb{{font-size:11px;color:#6b7280;margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em}}
  .mc .vl{{font-size:22px;font-weight:700;color:#111827}}
  .mc .sb{{font-size:11px;color:#6b7280;margin-top:2px}}
  .card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 20px;margin-bottom:16px}}
  .sec{{font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px}}
  table{{width:100%;border-collapse:collapse}}
  th{{text-align:left;color:#9ca3af;font-weight:600;font-size:10px;padding:6px 8px;border-bottom:1.5px solid #e5e7eb;text-transform:uppercase;letter-spacing:.05em;background:#fafafa}}
  .footer{{margin-top:36px;padding-top:14px;border-top:1px solid #e5e7eb;font-size:11px;color:#9ca3af;line-height:1.8}}
  @media(max-width:700px){{.g4{{grid-template-columns:1fr 1fr}}.g2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="page">

<h1>SPY — Analisis de Apertura · {today["date"]}</h1>
<div class="sub">Generado {datetime.now().strftime("%d/%m/%Y %H:%M")} &nbsp;·&nbsp; {res["base"]["n"]} dias historicos &nbsp;·&nbsp; Umbral: +/-{um}%{"&nbsp;·&nbsp; <strong>Precio de referencia personalizado</strong>" if today["precio_custom"] else ""}</div>

<div class="g4">
  <div class="mc" style="{"border-left:3px solid #f59e0b;" if today["precio_custom"] else ""}">
    <div class="lb">{"Precio de referencia" if today["precio_custom"] else "Precio de apertura"}</div>
    <div class="vl" style="{"color:#b45309;" if today["precio_custom"] else ""}">${today["precio_ref"]:.2f}</div>
    <div class="sb">{"open: $" + str(today["open"]) + " · personalizado" if today["precio_custom"] else today["date"]}</div>
  </div>
  <div class="mc"><div class="lb">Target alcista (+{um}%)</div><div class="vl" style="color:#16a34a">${today["target_up"]:.2f}</div><div class="sb">ref +{um}%</div></div>
  <div class="mc"><div class="lb">Target bajista (-{um}%)</div><div class="vl" style="color:#dc2626">${today["target_dn"]:.2f}</div><div class="sb">ref -{um}%</div></div>
  <div class="mc"><div class="lb">Contexto de hoy</div><div class="vl" style="font-size:14px">{GL.get(res["gap_bin"],"").split("(")[0].strip()}</div><div class="sb">{VL.get(res["vol_bin"],"")} · prev {today["prev"]:+.2f}%</div></div>
</div>

<div class="g2">
  <div>
    {_verdict_box(f"P(cierra >= +{um}% desde apertura)", kpp, bpp, ci_pp, up=True)}
    {_verdict_box(f"P(cierra <= -{um}% desde apertura)", kpm, bpm, ci_pm, up=False)}

    <div class="card">
      <div class="sec">Detalle de la estimacion k-NN (top-30 dias similares)</div>
      <div style="margin-bottom:12px">
        <div style="font-size:12px;color:#6b7280;margin-bottom:6px">P(cierra &gt;= +{um}%)</div>
        {_pbar(kpp, bpp, ci_pp, "#16a34a")}
      </div>
      <div style="margin-bottom:12px">
        <div style="font-size:12px;color:#6b7280;margin-bottom:6px">P(cierra &lt;= -{um}%)</div>
        {_pbar(kpm, bpm, ci_pm, "#dc2626")}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px">
        <div class="mc" style="text-align:center"><div class="lb">Ret. medio esperado</div>
          <div class="vl" style="font-size:20px;color:{"#16a34a" if res["knn"]["mr"]>0 else "#dc2626"}">{res["knn"]["mr"]:+.3f}%</div></div>
        <div class="mc" style="text-align:center"><div class="lb">Std. esperado</div>
          <div class="vl" style="font-size:20px">{res["knn"]["sr"]:.3f}%</div></div>
      </div>
    </div>

    <div class="card">
      <div class="sec">Top 5 dias mas similares al contexto de hoy</div>
      <table>
        <tr><th>Fecha</th><th style="text-align:center">Gap</th><th style="text-align:center">Prev</th><th style="text-align:center">Vol5</th>
            <th style="text-align:right">Ret final</th><th style="text-align:center">+{um}%</th><th style="text-align:center">-{um}%</th></tr>
        {top5}
      </table>
    </div>
  </div>

  <div>
    <div class="card">
      <div class="sec">Señales condicionales por variable</div>

      <div style="font-size:12px;color:#374151;font-weight:600;margin-bottom:4px">Gap de apertura</div>
      <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">Hoy: {today["gap"]:+.3f}% → {GL.get(res["gap_bin"],"")}</div>
      <table><tr><th>Condicion</th><th style="text-align:center">P(+{um}%)</th><th style="text-align:center">P(-{um}%)</th><th style="text-align:center">n</th></tr>{crow_gap}</table>

      <div style="margin-top:16px;font-size:12px;color:#374151;font-weight:600;margin-bottom:4px">Retorno del dia anterior</div>
      <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">Ayer: {today["prev"]:+.3f}% → {PL.get(res["prev_bin"],"")}</div>
      <table><tr><th>Condicion</th><th style="text-align:center">P(+{um}%)</th><th style="text-align:center">P(-{um}%)</th><th style="text-align:center">n</th></tr>{crow_prev}</table>

      <div style="margin-top:16px;font-size:12px;color:#374151;font-weight:600;margin-bottom:4px">Volatilidad reciente (std 5 dias)</div>
      <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">Vol5: {today["vol5"]:.4f}% · mediana hist: {res["vol_med"]:.4f}% → {VL.get(res["vol_bin"],"")}</div>
      <table><tr><th>Condicion</th><th style="text-align:center">P(+{um}%)</th><th style="text-align:center">P(-{um}%)</th><th style="text-align:center">n</th></tr>{crow_vol}</table>
    </div>

    <div class="card">
      <div class="sec">Percentiles del retorno esperado (open al cierre)</div>
      <table>
        <tr><th>Percentil</th><th style="text-align:right">Historia completa</th><th style="text-align:right">Top-30 similares</th></tr>
        {pct_rows}
      </table>
      <div style="margin-top:10px;font-size:11px;color:#9ca3af;line-height:1.6">
        Target alcista: <strong style="color:#16a34a">+{um}%</strong> = ${today["target_up"]:.2f} &nbsp;|&nbsp;
        Target bajista: <strong style="color:#dc2626">-{um}%</strong> = ${today["target_dn"]:.2f}
      </div>
    </div>
  </div>
</div>

<div class="card">
  <div class="sec">Distribucion historica de retornos desde la apertura al cierre</div>
  <div style="display:flex;gap:16px;margin-bottom:12px;font-size:12px;color:#6b7280;flex-wrap:wrap">
    <span style="display:flex;align-items:center;gap:5px"><span style="width:12px;height:12px;border-radius:2px;background:#93c5fd;display:inline-block"></span>Historia completa ({res["base"]["n"]} dias)</span>
    <span style="display:flex;align-items:center;gap:5px"><span style="width:12px;height:12px;border-radius:2px;background:#2563eb;display:inline-block"></span>Top-30 similares</span>
    <span style="display:flex;align-items:center;gap:5px"><span style="width:12px;height:12px;border-radius:2px;background:#86efac;display:inline-block"></span>Zona alcista (&gt;={um}%)</span>
    <span style="display:flex;align-items:center;gap:5px"><span style="width:12px;height:12px;border-radius:2px;background:#fca5a5;display:inline-block"></span>Zona bajista (&lt;=-{um}%)</span>
  </div>
  <div style="position:relative;width:100%;height:250px"><canvas id="distChart"></canvas></div>
  <div style="font-size:11px;color:#9ca3af;margin-top:8px;line-height:1.6">
    La distribucion de los dias similares (azul) suele ser mas estrecha que el historico completo. Las zonas sombreadas en verde/rojo son los targets de hoy.
    Base rate historica: P(+{um}%) = {round(bpp*100,1)}% · P(-{um}%) = {round(bpm*100,1)}%.
  </div>
</div>

<div class="footer">
  <strong>Razonamiento:</strong> en la apertura solo disponemos de 4 señales: gap (open vs close ayer), retorno del dia anterior, volatilidad reciente (std 5 dias) y momentum 3 dias. El modelo k-NN encuentra los 30 dias historicos mas parecidos en ese espacio (distancia euclidiana normalizada) y calcula empiricamente que fraccion supero cada umbral. Las señales condicionales por bin son auxiliares. El intervalo de confianza (IC 90%, Wilson) indica la precision de la estimacion — con n=30 es amplio, lo cual es honesto: predecir desde la apertura es dificil.
  <br><br>
  <strong>Aviso:</strong> analisis estadistico sobre patrones historicos. No es asesoramiento financiero. Los patrones pasados no garantizan resultados futuros.
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const bins={hbins}; const allC={hall}; const t30C={ht30}; const um={um};
const labels=bins.map(b=>b.toFixed(2)+'%');
const bgA=bins.map(b=>b>=um?'rgba(134,239,172,0.5)':b+0.25<=-um?'rgba(252,165,165,0.5)':'rgba(147,197,253,0.55)');
const bgT=bins.map(b=>b>=um?'rgba(22,163,74,0.9)':b+0.25<=-um?'rgba(220,38,38,0.9)':'rgba(37,99,235,0.85)');
new Chart(document.getElementById('distChart'),{{
  type:'bar',
  data:{{labels,datasets:[
    {{label:'Historia completa',data:allC,backgroundColor:bgA,borderRadius:2,borderSkipped:false,order:2}},
    {{label:'Top-30 similares',data:t30C,backgroundColor:bgT,borderRadius:2,borderSkipped:false,order:1}}
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}},
      tooltip:{{callbacks:{{
        title:c=>`Retorno ${{c[0].label}} a ${{(parseFloat(c[0].label)+0.25).toFixed(2)}}%`,
        label:c=>` ${{c.dataset.label}}: ${{c.parsed.y}} dias`
      }}}}
    }},
    scales:{{
      x:{{grid:{{display:false}},ticks:{{font:{{size:10}},color:'#9ca3af',maxRotation:45,
          callback:(_,i)=>{{const v=bins[i];return v%0.5===0?v.toFixed(1)+'%':'';}} }} }},
      y:{{beginAtZero:true,grid:{{color:'rgba(0,0,0,0.05)'}},
          ticks:{{font:{{size:11}},color:'#9ca3af'}},
          title:{{display:true,text:'Cantidad de dias',font:{{size:10}},color:'#9ca3af'}} }}
    }}
  }}
}});
</script>
</body>
</html>"""


# ── 4. MAIN ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analiza la apertura del SPY y predice probabilidad de alcanzar umbrales.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python spy_open_analysis.py SPY_1h_acum.csv
  python spy_open_analysis.py SPY_1h_acum.csv --precio 660.00
  python spy_open_analysis.py SPY_1h_acum.csv --precio 652.50 --umbral 0.75
  python spy_open_analysis.py SPY_1h_acum.csv --output apertura_hoy.html
        """
    )
    parser.add_argument("csv_1h", help="CSV acumulado con granularidad 1h (incluye hoy)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--umbral", type=float, default=0.5,
                        help="Umbral de precio en %% (default: 0.5)")
    parser.add_argument("--precio", type=float, default=None,
                        help="Precio de referencia personalizado (default: precio de apertura). "
                             "Ejemplo: --precio 660.00")
    args = parser.parse_args()

    if not Path(args.csv_1h).exists():
        print(f"ERROR: No se encontro '{args.csv_1h}'"); sys.exit(1)

    print(f"Cargando: {args.csv_1h}")
    hist, today = load_and_build(args.csv_1h, umbral=args.umbral, precio_ref=args.precio)
    print(f"  Dia: {today['date']} · Open: ${today['open']:.2f}")
    if today["precio_custom"]:
        print(f"  Precio de referencia personalizado: ${today['precio_ref']:.2f}  (diferencia vs open: {(today['precio_ref']/today['open']-1)*100:+.3f}%)")
    print(f"  Gap: {today['gap']:+.4f}% · Prev: {today['prev']:+.4f}% · Vol5: {today['vol5']:.4f}% · Ret3d: {today['ret3d']:+.4f}%")
    print(f"  {len(hist)} dias historicos")

    print(f"\nAnalizando (umbral +/-{args.umbral}% desde ref ${today['precio_ref']:.2f})...")
    res = run_analysis(hist, today)
    ci_pp=res["knn"]["ci_pp"]; ci_pm=res["knn"]["ci_pm"]
    print(f"  Base rate:  P(+{args.umbral}%)={res['base']['pp']:.1%}  P(-{args.umbral}%)={res['base']['pm']:.1%}")
    print(f"  k-NN top30: P(+{args.umbral}%)={res['knn']['pp']:.1%}  P(-{args.umbral}%)={res['knn']['pm']:.1%}")

    html = generate_html(today, res)
    ref_tag = f"_ref{str(today['precio_ref']).replace('.','p')}" if today["precio_custom"] else ""
    out = args.output or f"spy_open_{today['date'].replace('-','')}{ref_tag}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  Reporte: {out}")
    ref_label = f"ref ${today['precio_ref']:.2f}" + (" (personalizado)" if today["precio_custom"] else " (= open)")
    print(f"\n  ┌─ RESUMEN {today['date']} ──────────────────────────────────────────")
    print(f"  │  Open:               ${today['open']:.2f}")
    print(f"  │  Precio referencia:  ${today['precio_ref']:.2f}{'  ← personalizado' if today['precio_custom'] else '  (= open)'}")
    print(f"  │  Target alcista:     ${today['target_up']:.2f}  (+{args.umbral}% desde ref)")
    print(f"  │  Target bajista:     ${today['target_dn']:.2f}  (-{args.umbral}% desde ref)")
    print(f"  │")
    print(f"  │  P(cierra >= +{args.umbral}% desde ref):  {round(res['knn']['pp']*100,1)}%  IC=[{round(ci_pp[0]*100)}%–{round(ci_pp[1]*100)}%]")
    print(f"  │  P(cierra <= -{args.umbral}% desde ref):  {round(res['knn']['pm']*100,1)}%  IC=[{round(ci_pm[0]*100)}%–{round(ci_pm[1]*100)}%]")
    print(f"  │  Retorno esperado:          {res['knn']['mr']:+.3f}% +/- {res['knn']['sr']:.3f}%")
    print(f"  └────────────────────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()
