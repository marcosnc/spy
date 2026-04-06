"""
SPY Historical Data Downloader
================================
Descarga datos históricos del ETF SPY desde múltiples fuentes públicas gratuitas.

Fuentes disponibles:
  - yfinance     : Yahoo Finance (sin API key, recomendado)
  - alpha_vantage: Alpha Vantage (requiere API key gratuita)
  - polygon      : Polygon.io (requiere API key gratuita, tier free)

Granularidades disponibles:
  yfinance       : 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
  alpha_vantage  : 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
  polygon        : 1, 5, 15, 30, 60 (minutos), 1d, 1w, 1month

Instalación de dependencias:
  pip install yfinance pandas requests python-dotenv

Uso rápido:
  python spy_downloader.py
  python spy_downloader.py --source yfinance --interval 15m --period 6mo
  python spy_downloader.py --source yfinance --interval 1h --start 2024-01-01 --end 2024-12-31
  python spy_downloader.py --source alpha_vantage --interval 30min --apikey TU_KEY
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas no está instalado. Ejecutá: pip install pandas")
    sys.exit(1)


# ─────────────────────────────────────────────
# FUENTE 1: Yahoo Finance via yfinance
# ─────────────────────────────────────────────

def download_yfinance(interval: str, period: str = None,
                      start: str = None, end: str = None) -> pd.DataFrame:
    """
    Descarga datos de SPY usando yfinance (Yahoo Finance).
    
    Limitaciones de Yahoo Finance para datos intradiarios:
      - 1m  → máximo 7 días de historia
      - 2m, 5m, 15m, 30m, 90m → máximo 60 días
      - 60m, 1h → máximo 730 días (~2 años)
      - 1d en adelante → histórico completo

    Args:
        interval: Granularidad (ej: '15m', '1h', '1d')
        period:   Período relativo ('1d','5d','1mo','3mo','6mo','1y','2y','5y','max')
        start:    Fecha inicio 'YYYY-MM-DD' (alternativa a period)
        end:      Fecha fin   'YYYY-MM-DD' (alternativa a period)
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance no está instalado. Ejecutá: pip install yfinance")
        sys.exit(1)

    ticker = yf.Ticker("SPY")

    print(f"[yfinance] Descargando SPY | interval={interval} | ", end="")
    if period:
        print(f"period={period}")
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
    else:
        print(f"start={start} → end={end or 'hoy'}")
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if df.empty:
        print("ADVERTENCIA: No se recibieron datos. Verificá los parámetros o los límites de la API.")
        return df

    # Normalizar columnas
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.sort_index()

    print(f"[yfinance] ✓ {len(df)} velas descargadas  "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    return df


# ─────────────────────────────────────────────
# FUENTE 2: Alpha Vantage
# ─────────────────────────────────────────────

def download_alpha_vantage(interval: str, apikey: str,
                           months_back: int = 3) -> pd.DataFrame:
    """
    Descarga datos de SPY usando Alpha Vantage.
    API key gratuita en: https://www.alphavantage.co/support/#api-key

    Args:
        interval:   '1min','5min','15min','30min','60min','daily','weekly','monthly'
        apikey:     Tu API key de Alpha Vantage
        months_back: Cuántos meses hacia atrás descargar (intradiario: máx ~24 meses)
    """
    import requests

    BASE_URL = "https://www.alphavantage.co/query"
    intraday_intervals = {"1min", "5min", "15min", "30min", "60min"}

    all_frames = []

    if interval in intraday_intervals:
        # Descargar mes a mes (slice=year1month1..year2month12)
        slices = _get_av_slices(months_back)
        for sl in slices:
            params = {
                "function":   "TIME_SERIES_INTRADAY",
                "symbol":     "SPY",
                "interval":   interval,
                "outputsize": "full",
                "datatype":   "json",
                "extended_hours": "false",
                "slice":      sl,
                "apikey":     apikey,
            }
            print(f"[alpha_vantage] Descargando slice={sl} ...", end=" ", flush=True)
            r = requests.get(BASE_URL, params=params, timeout=30)
            data = r.json()

            key = f"Time Series ({interval})"
            if key not in data:
                print(f"⚠  Sin datos (respuesta: {list(data.keys())})")
                continue

            df_slice = pd.DataFrame(data[key]).T
            df_slice.index = pd.to_datetime(df_slice.index)
            print(f"✓ {len(df_slice)} filas")
            all_frames.append(df_slice)

    elif interval == "daily":
        params = {
            "function":   "TIME_SERIES_DAILY",
            "symbol":     "SPY",
            "outputsize": "full",
            "datatype":   "json",
            "apikey":     apikey,
        }
        print("[alpha_vantage] Descargando serie diaria completa ...", end=" ", flush=True)
        r = requests.get(BASE_URL, params=params, timeout=30)
        data = r.json()
        key = "Time Series (Daily)"
        if key not in data:
            print(f"ERROR: {data}")
            return pd.DataFrame()
        df_all = pd.DataFrame(data[key]).T
        df_all.index = pd.to_datetime(df_all.index)
        print(f"✓ {len(df_all)} filas")
        all_frames.append(df_all)

    else:
        func_map = {"weekly": "TIME_SERIES_WEEKLY", "monthly": "TIME_SERIES_MONTHLY"}
        key_map  = {"weekly": "Weekly Time Series", "monthly": "Monthly Time Series"}
        params = {
            "function": func_map[interval],
            "symbol":   "SPY",
            "datatype": "json",
            "apikey":   apikey,
        }
        print(f"[alpha_vantage] Descargando serie {interval} ...", end=" ", flush=True)
        r = requests.get(BASE_URL, params=params, timeout=30)
        data = r.json()
        key = key_map[interval]
        if key not in data:
            print(f"ERROR: {data}")
            return pd.DataFrame()
        df_all = pd.DataFrame(data[key]).T
        df_all.index = pd.to_datetime(df_all.index)
        print(f"✓ {len(df_all)} filas")
        all_frames.append(df_all)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames)
    df = df.rename(columns={
        "1. open": "open", "2. high": "high",
        "3. low":  "low",  "4. close": "close", "5. volume": "volume"
    })
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.index.name = "datetime"
    df = df.sort_index().drop_duplicates()

    # Filtrar al rango pedido
    cutoff = datetime.now() - timedelta(days=months_back * 30)
    df = df[df.index >= cutoff]

    print(f"[alpha_vantage] ✓ Total: {len(df)} velas  "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    return df


def _get_av_slices(months_back: int) -> list:
    """Genera los slice names de Alpha Vantage para los últimos N meses."""
    slices = []
    now = datetime.now()
    for i in range(months_back):
        d = now - timedelta(days=i * 30)
        year_offset  = (i // 12) + 1
        month_offset = (i % 12) + 1
        slices.append(f"year{year_offset}month{month_offset}")
    return slices[:min(months_back, 24)]  # AV permite hasta year2month12


# ─────────────────────────────────────────────
# FUENTE 3: Polygon.io
# ─────────────────────────────────────────────

def download_polygon(multiplier: int, timespan: str, apikey: str,
                     start: str = None, end: str = None,
                     months_back: int = 6) -> pd.DataFrame:
    """
    Descarga datos de SPY usando Polygon.io.
    API key gratuita en: https://polygon.io (plan Starter = datos con 15min delay)

    Args:
        multiplier: Factor numérico (ej: 15 para 15 minutos)
        timespan:   'minute','hour','day','week','month'
        apikey:     Tu API key de Polygon
        start:      'YYYY-MM-DD'
        end:        'YYYY-MM-DD'
        months_back: Si no se especifica start/end, cuántos meses hacia atrás
    """
    import requests

    if not start:
        start = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
    if not end:
        end = datetime.now().strftime("%Y-%m-%d")

    all_results = []
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/"
        f"{multiplier}/{timespan}/{start}/{end}"
    )
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   apikey,
    }

    print(f"[polygon] Descargando SPY | {multiplier} {timespan} | {start} → {end}")

    while url:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()

        if data.get("status") == "ERROR":
            print(f"ERROR Polygon: {data.get('error', data)}")
            break

        results = data.get("results", [])
        all_results.extend(results)
        print(f"  → {len(all_results)} velas acumuladas...", end="\r")

        # Paginación
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": apikey}
        else:
            break

    if not all_results:
        print("\n[polygon] Sin datos.")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("datetime")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].sort_index()

    print(f"\n[polygon] ✓ {len(df)} velas  "
          f"({df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    return df


# ─────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────

def save_data(df: pd.DataFrame, output_dir: str, interval: str, source: str):
    """Guarda el DataFrame en CSV y Parquet."""
    if df.empty:
        print("No hay datos para guardar.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"SPY_{interval}_{source}_{timestamp}"

    # CSV
    csv_path = os.path.join(output_dir, base_name + ".csv")
    df.to_csv(csv_path)
    print(f"✓ CSV guardado   → {csv_path}")

    # Parquet (más eficiente para grandes datasets)
    try:
        parquet_path = os.path.join(output_dir, base_name + ".parquet")
        df.to_parquet(parquet_path)
        print(f"✓ Parquet guardado → {parquet_path}")
    except ImportError:
        print("(Parquet omitido: instalá pyarrow con 'pip install pyarrow' para habilitarlo)")

    # Estadísticas rápidas
    print("\n─── Resumen del dataset ─────────────────────────────")
    print(f"  Filas      : {len(df):,}")
    print(f"  Columnas   : {list(df.columns)}")
    print(f"  Desde      : {df.index[0]}")
    print(f"  Hasta      : {df.index[-1]}")
    print(f"  Close min  : {df['close'].min():.2f}")
    print(f"  Close max  : {df['close'].max():.2f}")
    print(f"  Close last : {df['close'].iloc[-1]:.2f}")
    print("─────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
# ACTUALIZACIÓN ACUMULATIVA
# ─────────────────────────────────────────────

def _remove_last_line(path: Path):
    """Trunca el archivo eliminando la última línea no vacía."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell() - 1
        # saltar newlines finales
        while pos > 0:
            f.seek(pos)
            if f.read(1) != b"\n":
                break
            pos -= 1
        # retroceder hasta el newline anterior (inicio de la última línea)
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                break
            pos -= 1
    with open(path, "r+b") as f:
        f.truncate(pos + 1)  # conserva el \n del final de la penúltima línea


def update_cumulative(interval: str, output_dir: str):
    """
    Actualiza el archivo acumulativo para el intervalo dado.

    Lee el último timestamp del acumulativo existente, descarga los datos
    nuevos desde esa fecha (inclusive, para capturar velas del mismo día
    que quedaron incompletas), hace merge deduplicado y sobreescribe el
    acumulativo.

    El archivo debe seguir la convención de nombre:
        {output_dir}/SPY_{interval}_yfinance_cummulative.csv
    """
    interval_safe = interval.replace("/", "_")
    cumulative_path = Path(output_dir) / f"SPY_{interval_safe}_yfinance_cummulative.csv"

    if not cumulative_path.exists():
        print(f"ERROR: No se encontró el archivo acumulativo: {cumulative_path}")
        print("Ejecutá primero sin --update para crear el archivo base, "
              "luego renombralo a la convención: "
              f"SPY_{interval_safe}_yfinance_cummulative.csv")
        sys.exit(1)

    # Leer solo la columna de índice para obtener el último timestamp
    # sin tocar los valores numéricos (evita cambios de precisión float)
    idx_series = pd.read_csv(cumulative_path, index_col=0, usecols=[0]).index
    idx_series = pd.to_datetime(idx_series, utc=True).tz_convert("America/New_York")
    n_existing = len(idx_series)
    last_dt = idx_series.max()
    start_date = last_dt.strftime("%Y-%m-%d")

    print(f"Archivo acumulativo : {cumulative_path}")
    print(f"Velas existentes    : {n_existing:,}")
    print(f"Último registro     : {last_dt}")
    print(f"Descargando desde   : {start_date}\n")

    df_new = download_yfinance(interval=interval, start=start_date)

    if df_new.empty:
        print("No hay nuevos datos disponibles.")
        return

    # Convertir a NY timezone y quedarse con velas desde el último registro
    # (inclusive: la última barra puede haber sido incompleta al momento de descarga)
    df_new.index = df_new.index.tz_convert("America/New_York")
    df_append = df_new[df_new.index >= last_dt]

    if df_append.empty:
        print("No hay velas nuevas para agregar.")
        return

    # Eliminar la última línea del acumulativo para reemplazarla con datos frescos
    _remove_last_line(cumulative_path)

    # Append al CSV sin reescribir el archivo: preserva precisión original
    df_append.to_csv(cumulative_path, mode="a", header=False)

    print(f"\n✓ {len(df_append)} nuevas velas agregadas")
    print(f"✓ Acumulativo actualizado → {cumulative_path}  "
          f"({n_existing + len(df_append):,} velas totales)")
    print(f"  Nuevas velas: {df_append.index[0].strftime('%Y-%m-%d %H:%M')} → "
          f"{df_append.index[-1].strftime('%Y-%m-%d %H:%M')}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Descarga datos históricos del SPY con múltiples granularidades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Datos diarios de 2 años (sin API key)
  python spy_downloader.py --source yfinance --interval 1d --period 2y

  # 15 minutos, últimos 60 días
  python spy_downloader.py --source yfinance --interval 15m --period 60d

  # 1 hora, rango de fechas específico
  python spy_downloader.py --source yfinance --interval 1h --start 2024-01-01 --end 2024-12-31

  # Actualizar acumulativo de 1h con los datos nuevos desde el último registro
  python spy_downloader.py --interval 1h --update

  # Actualizar acumulativo de 15m
  python spy_downloader.py --interval 15m --update

  # Alpha Vantage, 30 minutos, 6 meses
  python spy_downloader.py --source alpha_vantage --interval 30min --apikey TU_KEY --months 6

  # Polygon, 15 minutos, últimos 3 meses
  python spy_downloader.py --source polygon --multiplier 15 --timespan minute --apikey TU_KEY --months 3
        """
    )

    parser.add_argument("--source", choices=["yfinance", "alpha_vantage", "polygon"],
                        default="yfinance",
                        help="Fuente de datos (default: yfinance)")

    # yfinance args
    parser.add_argument("--interval", default="1h",
                        help="Granularidad para yfinance/alpha_vantage (ej: 15m, 1h, 1d)")
    parser.add_argument("--period", default=None,
                        help="Período relativo yfinance (ej: 6mo, 1y, 2y)")
    parser.add_argument("--start", default=None, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="Fecha fin   YYYY-MM-DD")

    # polygon args
    parser.add_argument("--multiplier", type=int, default=15,
                        help="Multiplicador para Polygon (ej: 15 para 15 minutos)")
    parser.add_argument("--timespan", default="minute",
                        choices=["minute", "hour", "day", "week", "month"],
                        help="Unidad de tiempo para Polygon")

    # comunes
    parser.add_argument("--apikey", default=None,
                        help="API key para Alpha Vantage o Polygon")
    parser.add_argument("--months", type=int, default=6,
                        help="Meses hacia atrás (para alpha_vantage y polygon)")
    parser.add_argument("--output", default="./spy_data",
                        help="Directorio de salida (default: ./spy_data)")
    parser.add_argument("--update", action="store_true",
                        help="Actualiza el acumulativo existente con datos nuevos "
                             "(solo yfinance). Lee el último timestamp del archivo "
                             "SPY_{interval}_yfinance_cummulative.csv, descarga lo "
                             "que falta y hace merge deduplicado.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n══════════════════════════════════════════════")
    print("       SPY Historical Data Downloader")
    print("══════════════════════════════════════════════\n")

    if args.update:
        if args.source != "yfinance":
            print("ERROR: --update solo está soportado con --source yfinance")
            sys.exit(1)
        update_cumulative(interval=args.interval, output_dir=args.output)
        return

    df = pd.DataFrame()

    if args.source == "yfinance":
        if not args.period and not args.start:
            args.period = "6mo"  # default razonable
        df = download_yfinance(
            interval=args.interval,
            period=args.period,
            start=args.start,
            end=args.end
        )

    elif args.source == "alpha_vantage":
        if not args.apikey:
            print("ERROR: Alpha Vantage requiere --apikey")
            print("Obtené una gratis en: https://www.alphavantage.co/support/#api-key")
            sys.exit(1)
        df = download_alpha_vantage(
            interval=args.interval,
            apikey=args.apikey,
            months_back=args.months
        )

    elif args.source == "polygon":
        if not args.apikey:
            print("ERROR: Polygon requiere --apikey")
            print("Obtené una gratis en: https://polygon.io")
            sys.exit(1)
        df = download_polygon(
            multiplier=args.multiplier,
            timespan=args.timespan,
            apikey=args.apikey,
            start=args.start,
            end=args.end,
            months_back=args.months
        )

    save_data(df, args.output, args.interval.replace("/", "_"), args.source)


if __name__ == "__main__":
    main()
