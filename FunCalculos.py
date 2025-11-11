import io
import pandas as pd
import numpy as np
from datetime import timedelta
import calendar
import warnings
from typing import Dict, Any, List, Optional, Union

# Suprimir warnings específicos do pandas
warnings.filterwarnings('ignore', category=FutureWarning)

def clean_numeric_value(value):
    """Converte valores numéricos brasileiros para float seguro."""
    if pd.isna(value) or value == '':
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    str_value = str(value).strip()
    if str_value == '':
        return np.nan

    # Remover separador de milhares (.)
    if '.' in str_value and ',' in str_value:
        integer_part, decimal_part = str_value.rsplit(',', 1)
        integer_part = integer_part.replace('.', '')
        cleaned = f"{integer_part}.{decimal_part}"
    elif ',' in str_value:
        cleaned = str_value.replace('.', '').replace(',', '.')
    else:
        # Pode ter múltiplos pontos como separador de milhar
        parts = str_value.split('.')
        if len(parts) > 1 and all(part.isdigit() for part in parts[1:]):
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        else:
            cleaned = str_value

    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def _safe_mean(series, default=0.0, absolute: bool = False):
    """Retorna média segura (sem NaN) com opção de valor absoluto."""
    if series is None:
        return default
    series = series.dropna()
    if series.empty:
        return default
    mean_value = series.mean()
    if pd.isna(mean_value):
        return default
    mean_value = float(mean_value)
    return abs(mean_value) if absolute else mean_value

def _normalize_trades_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma cópia do DataFrame com as colunas padrão utilizadas nos cálculos,
    independentemente do layout original do CSV.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()

    # Datas de entrada e saída
    if 'entry_date' not in normalized.columns and 'Abertura' in normalized.columns:
        normalized['entry_date'] = pd.to_datetime(normalized['Abertura'], errors='coerce')
    else:
        normalized['entry_date'] = pd.to_datetime(normalized.get('entry_date'), errors='coerce')

    if 'exit_date' not in normalized.columns and 'Fechamento' in normalized.columns:
        normalized['exit_date'] = pd.to_datetime(normalized['Fechamento'], errors='coerce')
    else:
        normalized['exit_date'] = pd.to_datetime(normalized.get('exit_date'), errors='coerce')

    # Resultado financeiro
    if 'pnl' not in normalized.columns:
        for col in ['operation_result', 'Res. Operação', 'Res. Intervalo', 'Resultado', 'Resultado Operação']:
            if col in normalized.columns:
                normalized['pnl'] = pd.to_numeric(normalized[col], errors='coerce')
                break
    else:
        normalized['pnl'] = pd.to_numeric(normalized['pnl'], errors='coerce')

    if 'pnl' not in normalized.columns:
        normalized['pnl'] = 0.0

    # Percentual
    if 'pnl_pct' not in normalized.columns:
        for col in ['operation_result_pct', 'Res. Operação (%)', 'Res. Intervalo (%)']:
            if col in normalized.columns:
                normalized['pnl_pct'] = pd.to_numeric(normalized[col], errors='coerce')
                break

    # Preços
    if 'entry_price' not in normalized.columns and 'Preço Compra' in normalized.columns:
        normalized['entry_price'] = pd.to_numeric(normalized['Preço Compra'], errors='coerce')
    elif 'entry_price' in normalized.columns:
        normalized['entry_price'] = pd.to_numeric(normalized['entry_price'], errors='coerce')

    if 'exit_price' not in normalized.columns and 'Preço Venda' in normalized.columns:
        normalized['exit_price'] = pd.to_numeric(normalized['Preço Venda'], errors='coerce')
    elif 'exit_price' in normalized.columns:
        normalized['exit_price'] = pd.to_numeric(normalized['exit_price'], errors='coerce')

    # Quantidades
    if 'qty_buy' not in normalized.columns and 'Qtd Compra' in normalized.columns:
        normalized['qty_buy'] = pd.to_numeric(normalized['Qtd Compra'], errors='coerce')
    elif 'qty_buy' in normalized.columns:
        normalized['qty_buy'] = pd.to_numeric(normalized['qty_buy'], errors='coerce')

    if 'qty_sell' not in normalized.columns and 'Qtd Venda' in normalized.columns:
        normalized['qty_sell'] = pd.to_numeric(normalized['Qtd Venda'], errors='coerce')
    elif 'qty_sell' in normalized.columns:
        normalized['qty_sell'] = pd.to_numeric(normalized['qty_sell'], errors='coerce')

    candidate_quantity_cols = [
        'quantity_total', 'quantity', 'Contracts', 'contracts', 'Quantidade', 'Qtd Contratos', 'Qtd', 'position', 'Position', 'Qtd Total'
    ]

    position_candidates = []
    for col in ['qty_buy', 'qty_sell']:
        if col in normalized.columns:
            position_candidates.append(normalized[col].abs())

    for col in candidate_quantity_cols:
        if col in normalized.columns:
            position_candidates.append(pd.to_numeric(normalized[col], errors='coerce').abs())

    if position_candidates:
        position_df = pd.concat(position_candidates, axis=1)
        normalized['position_size'] = position_df.max(axis=1, skipna=True)
    else:
        normalized['position_size'] = np.nan

    # Se nenhuma quantidade foi encontrada, assumir 1 contrato por trade
    normalized['position_size'] = normalized['position_size'].fillna(1).astype(float)

    # Direção e ativo
    if 'direction' not in normalized.columns and 'Lado' in normalized.columns:
        normalized['direction'] = normalized['Lado']
    if 'symbol' not in normalized.columns and 'Ativo' in normalized.columns:
        normalized['symbol'] = normalized['Ativo']

    # Duração em minutos
    if 'entry_date' in normalized.columns and 'exit_date' in normalized.columns:
        durations = (normalized['exit_date'] - normalized['entry_date']).dt.total_seconds() / 60
        normalized['duration_minutes'] = durations.clip(lower=0).fillna(0)
    else:
        normalized['duration_minutes'] = 0.0

    return normalized

def _format_minutes(value: float) -> str:
    """Converte minutos em string humanizada (Hh Mm)."""
    if pd.isna(value):
        value = 0
    total_minutes = max(0, float(value))
    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

def calcular_dimensionamento_posicao(df: pd.DataFrame) -> Dict[str, Any]:
    normalized = _normalize_trades_dataframe(df)
    if normalized.empty:
        return {
            "hasData": False,
            "average": 0.0,
            "median": 0.0,
            "maximum": 0.0,
            "dailyTurnover": 0.0,
            "distribution": []
        }

    qty_series = normalized['position_size'].dropna()
    if qty_series.empty:
        return {
            "hasData": False,
            "average": 0.0,
            "median": 0.0,
            "maximum": 0.0,
            "dailyTurnover": 0.0,
            "distribution": []
        }

    qty_series = qty_series.astype(float)
    average = qty_series.mean()
    median = qty_series.median()
    maximum = qty_series.max()

    if 'entry_date' in normalized.columns:
        normalized['date_only'] = pd.to_datetime(normalized['entry_date']).dt.date
        turnover = normalized.dropna(subset=['position_size']).groupby('date_only')['position_size'].sum()
        daily_turnover = turnover.mean() if not turnover.empty else average
    else:
        daily_turnover = average

    distribution = []
    grouped = normalized.dropna(subset=['position_size']).groupby('position_size')
    total_trades = len(qty_series)

    for pos_size, sub_df in grouped:
        pnl = pd.to_numeric(sub_df.get('pnl', 0), errors='coerce').fillna(0)
        trades = len(sub_df)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
        win_rate = len(wins) / trades * 100 if trades > 0 else 0.0
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        payoff = avg_win / avg_loss if avg_loss else 0.0

        distribution.append({
            "contracts": float(pos_size),
            "trades": trades,
            "percent": round(trades / total_trades * 100, 2) if total_trades > 0 else 0.0,
            "profitFactor": round(profit_factor, 2),
            "winRate": round(win_rate, 2),
            "payoff": round(payoff, 2),
            "result": round(float(pnl.sum()), 2)
        })

    distribution.sort(key=lambda x: (-x["trades"], -x["contracts"]))

    return {
        "hasData": True,
        "average": round(float(average), 2),
        "median": round(float(median), 2),
        "maximum": round(float(maximum), 2),
        "dailyTurnover": round(float(daily_turnover), 2),
        "distribution": distribution
    }

def calcular_duracao_trades_resumo(df: pd.DataFrame) -> Dict[str, Any]:
    normalized = _normalize_trades_dataframe(df)
    if normalized.empty or 'duration_minutes' not in normalized.columns:
        return {
            "hasData": False,
            "averageMinutes": 0.0,
            "medianMinutes": 0.0,
            "maxMinutes": 0.0,
            "average": "0m",
            "median": "0m",
            "maximum": "0m",
            "distribution": []
        }

    durations = normalized['duration_minutes'].dropna()
    if durations.empty:
        return {
            "hasData": False,
            "averageMinutes": 0.0,
            "medianMinutes": 0.0,
            "maxMinutes": 0.0,
            "average": "0m",
            "median": "0m",
            "maximum": "0m",
            "distribution": []
        }

    avg = durations.mean()
    med = durations.median()
    mx = durations.max()

    # Distribuição simples por faixas de duração
    bins = [0, 15, 30, 60, 120, np.inf]
    labels = ["0-15m", "15-30m", "30-60m", "60-120m", ">120m"]
    normalized['duration_bucket'] = pd.cut(durations, bins=bins, labels=labels, include_lowest=True)
    normalized['duration_bucket'] = normalized['duration_bucket'].astype(pd.CategoricalDtype(categories=labels, ordered=True))
    distribution = []
    bucket_group = normalized.groupby('duration_bucket', observed=True)

    for label, sub in bucket_group:
        if sub is None or sub.empty:
            continue
        pnl = pd.to_numeric(sub.get('pnl', 0), errors='coerce').fillna(0)
        trades = len(sub)
        gross_profit = pnl[pnl > 0].sum()
        gross_loss = abs(pnl[pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
        win_rate = (pnl > 0).sum() / trades * 100 if trades > 0 else 0.0
        avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0.0
        avg_loss = abs(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0
        payoff = avg_win / avg_loss if avg_loss else 0.0

        distribution.append({
            "bucket": label,
            "trades": trades,
            "profitFactor": round(profit_factor, 2),
            "winRate": round(win_rate, 2),
            "payoff": round(payoff, 2),
            "result": round(float(pnl.sum()), 2)
        })

    return {
        "hasData": True,
        "averageMinutes": round(float(avg), 2),
        "medianMinutes": round(float(med), 2),
        "maxMinutes": round(float(mx), 2),
        "average": _format_minutes(avg),
        "median": _format_minutes(med),
        "maximum": _format_minutes(mx),
        "distribution": distribution
    }

def calcular_custos_backtest(df: pd.DataFrame, taxa_corretagem: float = 0.0, taxa_emolumentos: float = 0.0) -> Dict[str, Any]:
    normalized = _normalize_trades_dataframe(df)
    if normalized.empty:
        return {
            "hasData": False,
            "totalTrades": 0,
            "valorOperado": 0.0,
            "corretagem": 0.0,
            "emolumentos": 0.0,
            "custoTotal": 0.0,
            "custoPorTrade": 0.0
        }

    if 'entry_price' not in normalized.columns or 'exit_price' not in normalized.columns:
        return {
            "hasData": False,
            "totalTrades": int(len(normalized)),
            "valorOperado": 0.0,
            "corretagem": 0.0,
            "emolumentos": 0.0,
            "custoTotal": 0.0,
            "custoPorTrade": 0.0
        }

    df_valid = normalized.dropna(subset=['entry_price', 'exit_price'])
    total_trades = len(df_valid)
    if total_trades == 0:
        return {
            "hasData": False,
            "totalTrades": 0,
            "valorOperado": 0.0,
            "corretagem": 0.0,
            "emolumentos": 0.0,
            "custoTotal": 0.0,
            "custoPorTrade": 0.0
        }

    valor_entrada = df_valid['entry_price'] * df_valid['position_size']
    valor_saida = df_valid['exit_price'] * df_valid['position_size']
    valor_total_operado = float((valor_entrada + valor_saida).sum())

    custo_corretagem = total_trades * taxa_corretagem
    custo_emolumentos = valor_total_operado * (taxa_emolumentos / 100.0)
    custo_total = custo_corretagem + custo_emolumentos

    return {
        "hasData": True,
        "totalTrades": int(total_trades),
        "valorOperado": round(valor_total_operado, 2),
        "corretagem": round(custo_corretagem, 2),
        "emolumentos": round(custo_emolumentos, 2),
        "custoTotal": round(custo_total, 2),
        "custoPorTrade": round(custo_total / total_trades, 2) if total_trades > 0 else 0.0
    }

def _read_csv_with_header(file_obj):
    """
    Lê conteúdo do CSV e garante que o cabeçalho real (linha com 'Ativo;') seja usado.
    Retorna DataFrame com os dados brutos.
    """
    try:
        if hasattr(file_obj, 'read'):
            original_position = file_obj.tell()
            raw_content = file_obj.read()
            if isinstance(raw_content, bytes):
                raw_content = raw_content.decode('latin1', errors='ignore')
            file_obj.seek(original_position)
        else:
            with open(file_obj, 'rb') as fh:
                raw_content = fh.read().decode('latin1', errors='ignore')
    except Exception as exc:
        raise ValueError(f"Erro ao ler o arquivo CSV: {exc}")

    lines = raw_content.splitlines()
    header_idx = next((idx for idx, line in enumerate(lines) if 'Ativo;' in line), None)
    if header_idx is None:
        raise ValueError("Cabeçalho 'Ativo;...' não encontrado no CSV.")

    csv_text = '\n'.join(lines[header_idx:])
    data_buffer = io.StringIO(csv_text)
    return pd.read_csv(data_buffer, sep=';', decimal=',', encoding='latin1')

def _compute_equity_components(pnl_series: pd.Series):
    """
    Calcula curvas de equity, pico e drawdown com baseline iniciando em 0.
    Retorna (equity, peak, drawdown, max_drawdown, max_drawdown_pct).
    """
    if pnl_series is None or pnl_series.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, 0.0, 0.0

    pnl_series = pnl_series.fillna(0)
    equity = pnl_series.cumsum()
    equity_with_start = pd.concat([pd.Series([0.0], index=[-1]), equity])
    peak = equity_with_start.cummax()
    drawdown = equity_with_start - peak

    # Remover ponto inicial artificial
    equity = equity_with_start.iloc[1:]
    peak = peak.iloc[1:]
    drawdown = drawdown.iloc[1:]

    if drawdown.empty:
        return equity, peak, drawdown, 0.0, 0.0

    max_dd = float(abs(drawdown.min()))
    peak_max = peak.max()
    max_dd_pct = float((max_dd / peak_max * 100) if peak_max not in (0, np.nan) else 0.0)

    return equity, peak, drawdown, max_dd, max_dd_pct

def carregar_csv(file):
    try:
        df = _read_csv_with_header(file)

        if 'Abertura' in df.columns:
            df['Abertura'] = pd.to_datetime(df['Abertura'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        if 'Fechamento' in df.columns:
            df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')

        numeric_columns = [
            'Res. Operação', 'Res. Operação (%)',
            'Res. Intervalo', 'Res. Intervalo (%)',
            'Res. Intervalo Bruto', 'Res. Intervalo Bruto (%)',
            'Preço Compra', 'Preço Venda', 'Preço de Mercado', 'Médio',
            'Qtd Compra', 'Qtd Venda', 'Drawdown', 'Ganho Max.', 'Perda Max.', 'Total'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)

        return df
    except Exception as e:
        raise ValueError(f"Erro ao processar CSV: {e}")

def _format_timedelta(td: timedelta, portuguese: bool = True, keep_seconds: bool = True):
    days = td.days
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if portuguese:
        parts = []
        if days > 0:
            parts.append(f"{days} dias")
        if hours > 0:
            parts.append(f"{hours} horas")
        if minutes > 0 or (days == 0 and hours == 0):
            parts.append(f"{minutes} min")
        if len(parts) > 1:
            return ' e '.join([', '.join(parts[:-1]), parts[-1]]) if len(parts) > 2 else f'{parts[0]} e {parts[1]}'
        elif parts:
            return parts[0]
        else:
            return "0 min"
    else:
        return str(td)

def calcular_metrics(sub, cdi=0.12):
    # Determinar qual coluna usar para PnL
    if 'operation_result' in sub.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in sub.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in sub.columns:
        pnl_col = 'Res. Operação'
    else:
        # Fallback para primeira coluna numérica
        numeric_cols = sub.select_dtypes(include=[np.number]).columns
        pnl_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if pnl_col is None:
        return 0, 0, 0, 0, 0
    
    pnl = sub[pnl_col]
    lucro = pnl.sum()
    trades = len(sub)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    returns = pnl.values
    mean_return = np.mean(returns) if trades > 0 else 0
    std_return = np.std(returns, ddof=1) if trades > 1 else 0
    sharpe = (mean_return - cdi) / std_return if std_return > 0 else 0

    equity = pnl.cumsum()
    peak = equity.cummax()
    dd_ser = equity - peak
    max_dd = float(dd_ser.min()) if not dd_ser.empty else 0
    sharpe_dd_simplificado = ((lucro / (3 * abs(max_dd))) - cdi) * 3 if max_dd != 0 else 0

    return lucro, pf, sharpe, sharpe_dd_simplificado, trades

def calcular_performance(df, cdi=0.12):
    total_trades = len(df)
    
    # Determinar qual coluna usar para PnL
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        # Fallback para primeira coluna numérica
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        pnl_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if pnl_col is None:
        return {}
    
    pnl = df[pnl_col]
    net_profit = pnl.sum()
    profit_trades = df[pnl > 0]
    loss_trades = df[pnl < 0]
    gross_profit = profit_trades[pnl_col].sum()
    gross_loss = abs(loss_trades[pnl_col].sum())
    avg_win = _safe_mean(profit_trades[pnl_col])
    avg_loss = _safe_mean(loss_trades[pnl_col], absolute=True)
    avg_per_trade = net_profit / total_trades if total_trades > 0 else 0
    win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0
    payoff = avg_win / abs(avg_loss) if avg_loss != 0 else 0

    returns = pnl.dropna().values
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
    sharpe_ratio = ((mean_return - cdi) / std_return) if std_return != 0 else 0
    # Determinar coluna de data
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        date_col = None
    
    if date_col is None:
        dias_com_operacoes = 1
    else:
        dias_com_operacoes = df[date_col].dt.date.nunique()
    media_operacoes_por_dia = total_trades / dias_com_operacoes if dias_com_operacoes > 0 else 0

    # PADRONIZADO: Calcular drawdown usando método centralizado
    equity, peak, dd_ser, max_dd, pct_dd = _compute_equity_components(pnl)
    
    # CALCULAR DD MÉDIO - CORREÇÃO ADICIONADA
    # Calcular drawdown médio baseado nos trades individuais
    drawdown_values = dd_ser[dd_ser < 0].abs()  # Apenas valores negativos (drawdowns)
    avg_drawdown = drawdown_values.mean() if len(drawdown_values) > 0 else 0

    recovery = net_profit / abs(max_dd) if max_dd != 0 else None
    recovery_3x = net_profit / (3 * abs(max_dd)) if max_dd != 0 else None
    sharpe_dd_simplificado = ((net_profit / (3 * abs(max_dd))) - cdi) * 3 if max_dd != 0 else 0

    cur_loss_sum = 0.0
    max_seq_loss = 0.0
    for x in pnl:
        if x < 0:
            cur_loss_sum += x
            max_seq_loss = min(max_seq_loss, cur_loss_sum)
        else:
            cur_loss_sum = 0.0
    trade_dd = max_seq_loss
    ending_equity = equity.iloc[-1] if len(equity) > 0 else 0
    trade_dd_pct = abs(trade_dd) / ending_equity * 100 if ending_equity != 0 else 0

    max_seq_win = max_seq_loss_cnt = 0
    cw = cl = 0
    for x in pnl:
        if x > 0:
            cw += 1
            cl = 0
            max_seq_win = max(max_seq_win, cw)
        elif x < 0:
            cl += 1
            cw = 0
            max_seq_loss_cnt = max(max_seq_loss_cnt, cl)

    # Determinar colunas de data
    if 'entry_date' in df.columns and 'exit_date' in df.columns:
        entry_col = 'entry_date'
        exit_col = 'exit_date'
    elif 'Abertura' in df.columns and 'Fechamento' in df.columns:
        entry_col = 'Abertura'
        exit_col = 'Fechamento'
    else:
        entry_col = exit_col = None
    
    if entry_col and exit_col:
        durations = df[exit_col] - df[entry_col]
        avg_dur = durations.mean() if total_trades > 0 else timedelta(0)
        dur_win = (profit_trades[exit_col] - profit_trades[entry_col]).mean() if len(profit_trades) > 0 else timedelta(0)
        dur_loss = (loss_trades[exit_col] - loss_trades[entry_col]).mean() if len(loss_trades) > 0 else timedelta(0)
    else:
        avg_dur = dur_win = dur_loss = timedelta(0)
    
    # CALCULAR DIAS VENCEDORES E PERDEDORES
    # Agrupar por dia e calcular se o dia foi vencedor ou perdedor
    if entry_col:
        df['date'] = df[entry_col].dt.date
        daily_pnl = df.groupby('date')[pnl_col].sum()
        winning_days = len(daily_pnl[daily_pnl > 0])
        losing_days = len(daily_pnl[daily_pnl < 0])
    else:
        winning_days = 0
        losing_days = 0

    max_trade_gain = pnl.max()
    max_trade_loss = pnl.min()

    return {
        "Total Trades": total_trades,
        "Net Profit": net_profit,
        "Gross Profit": gross_profit,
        "Gross Loss": gross_loss,
        "Profit Factor": profit_factor,
        "Payoff": payoff,
        "Win Rate (%)": win_rate,
        "Average PnL/Trade": avg_per_trade,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Max Trade Gain": max_trade_gain,
        "Max Trade Loss": max_trade_loss,
        "Max Consecutive Wins": max_seq_win,
        "Max Consecutive Losses": max_seq_loss_cnt,
        "Max Drawdown ($)": max_dd,
        "Max Drawdown (%)": pct_dd,
        "Average Drawdown ($)": avg_drawdown,  # NOVO: DD Médio
        "Max Drawdown Padronizado ($)": max_dd,  # Valor padronizado
        "Max Drawdown Padronizado (%)": pct_dd,  # Percentual padronizado
        "Max Trade Drawdown ($)": trade_dd,
        "Max Trade Drawdown (%)": trade_dd_pct,
        "Average Trade Duration": _format_timedelta(avg_dur),
        "Avg Winning Trade Duration": _format_timedelta(dur_win),
        "Avg Losing Trade Duration": _format_timedelta(dur_loss),
        "Recovery Factor": recovery,
        "Sharpe Ratio": sharpe_dd_simplificado,
        "Avg Trades/Active Day": media_operacoes_por_dia,
        "Active Days": dias_com_operacoes,
        "Winning Days": winning_days,
        "Losing Days": losing_days,
    }

def calcular_day_of_week(df, cdi=0.12):
    df = df.copy()
    
    # Determinar coluna de data
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        return {}
    
    # Determinar coluna de PnL
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        return {}
    
    df['DayOfWeek'] = df[date_col].dt.day_name()
    dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    stats = {}
    for dia in dias:
        sub = df[df['DayOfWeek'] == dia]
        lucro, pf, sharpe, sharpe_simp, trades = calcular_metrics(sub, cdi=cdi)
        wins = sub[sub[pnl_col] > 0]
        losses = sub[sub[pnl_col] < 0]
        win_rate = len(wins) / trades * 100 if trades > 0 else 0
        
        # Calcular média de ganho e perda
        avg_win = _safe_mean(wins[pnl_col])
        avg_loss = _safe_mean(losses[pnl_col], absolute=True)
        
        # Calcular rentabilidade total (lucro total do período)
        rentabilidade_total = lucro
        
        stats[dia] = {
            "Trades": trades,
            "Net Profit": lucro,
            "Profit Factor": pf,
            "Win Rate (%)": round(win_rate, 2),
            "Sharpe Ratio": sharpe,
            "Average Win": round(avg_win, 2),
            "Average Loss": round(avg_loss, 2),
            "Rentabilidade ($)": round(rentabilidade_total, 2)
        }

    dias_com_operacoes = {k: v for k, v in stats.items() if v["Trades"] > 0}

    if dias_com_operacoes:
        best_day = max(dias_com_operacoes.items(), key=lambda x: x[1]['Net Profit'])[0]
        worst_day = min(dias_com_operacoes.items(), key=lambda x: x[1]['Net Profit'])[0]
        worst_day_stats = stats.get(worst_day, {})
        best_day_stats = stats.get(best_day, {})
    else:
        best_day = worst_day = 0
        worst_day_stats = best_day_stats = {
            "Trades": 0,
            "Net Profit": 0,
            "Profit Factor": 0,
            "Win Rate (%)": 0,
            "Sharpe Ratio": 0,
            "Rentabilidade ($)": 0
        }

    return {
        "Stats": stats,
        "Best Day": {"Day": best_day, **best_day_stats},
        "Worst Day": {"Day": worst_day, **worst_day_stats}
    }

def calcular_monthly(df, cdi=0.12):
    df = df.copy()
    
    # Determinar colunas
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        return {}
    
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        return {}
    
    # Ordenar por data de abertura para cálculo correto do drawdown
    df = df[df[date_col].notna()].copy()
    if df.empty:
        start_ts = pd.Timestamp.utcnow()
        return [{
            "date": start_ts.strftime('%Y-%m-%d'),
            "fullDate": start_ts.strftime('%d/%m/%Y'),
            "saldo": 0.0,
            "valor": float(capital_inicial),
            "resultado": 0.0,
            "drawdown": 0.0,
            "drawdownPercent": 0.0,
            "peak": float(capital_inicial),
            "trades": 0,
            "isStart": True
        }]

    df = df[df[date_col].notna()].copy()
    if df.empty:
        start_ts = pd.Timestamp.utcnow()
        return [{
            "date": start_ts.strftime('%Y-%m-%d'),
            "fullDate": start_ts.strftime('%d/%m/%Y'),
            "saldo": 0.0,
            "valor": float(capital_inicial),
            "resultado": 0.0,
            "drawdown": 0.0,
            "drawdownPercent": 0.0,
            "peak": float(capital_inicial),
            "trades": 0,
            "resultado_periodo": 0.0,
            "periodo": agrupar_por,
            "isStart": True
        }]

    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calcular equity curve global para drawdown correto
    df['Saldo'] = df[pnl_col].cumsum()
    df['Saldo_Maximo'] = df['Saldo'].cummax()
    df['Drawdown'] = df['Saldo'] - df['Saldo_Maximo']
    
    df['MonthNum'] = df[date_col].dt.month
    df['Year'] = df[date_col].dt.year
    df['YearMonth'] = df[date_col].dt.to_period('M')
    
    stats = {}

    for m in range(1, 13):
        sub = df[df['MonthNum'] == m]
        
        if len(sub) > 0:
            lucro, pf, sharpe, sharpe_simp, trades = calcular_metrics(sub, cdi=cdi)

            # Taxa de acerto
            wins = sub[sub[pnl_col] > 0]
            losses = sub[sub[pnl_col] < 0]
            win_rate = len(wins) / trades * 100 if trades > 0 else 0
            
            # Calcular média de ganho e perda
            avg_win = _safe_mean(wins[pnl_col])
            avg_loss = _safe_mean(losses[pnl_col], absolute=True)
            
            # Calcular drawdown máximo do mês
            max_drawdown_mes = sub['Drawdown'].min() if not sub['Drawdown'].empty else 0
            
            # Calcular drawdown percentual baseado no saldo final do mês
            saldo_final_mes = sub['Saldo'].iloc[-1] if not sub['Saldo'].empty else 0
            drawdown_percentual = (abs(max_drawdown_mes) / saldo_final_mes * 100) if saldo_final_mes != 0 else 0

            # Calcular rentabilidade total (lucro total do período)
            rentabilidade_total = lucro

            stats[calendar.month_name[m]] = {
                "Trades": trades,
                "Win Rate (%)": round(win_rate, 2),
                "Net Profit": lucro,
                "Profit Factor": pf,
                "Sharpe Ratio": sharpe,
                "Average Win": round(avg_win, 2),
                "Average Loss": round(avg_loss, 2),
                "Max Drawdown ($)": round(max_drawdown_mes, 2),
                "Max Drawdown (%)": round(drawdown_percentual, 2),
                "Rentabilidade ($)": round(rentabilidade_total, 2)
            }
        else:
            stats[calendar.month_name[m]] = {
                "Trades": 0,
                "Win Rate (%)": 0,
                "Net Profit": 0.0,
                "Profit Factor": 0,
                "Sharpe Ratio": 0,
                "Max Drawdown ($)": 0.0,
                "Max Drawdown (%)": 0.0,
                "Rentabilidade ($)": 0.0
            }

    # Filtrar apenas meses com operações para determinar melhor/pior
    meses_com_operacoes = {k: v for k, v in stats.items() if v["Trades"] > 0}
    
    if meses_com_operacoes:
        best_month = max(meses_com_operacoes.items(), key=lambda x: x[1]['Net Profit'])[0]
        worst_month = min(
            (item for item in meses_com_operacoes.items() if item[1]['Net Profit'] < 0),
            key=lambda x: x[1]['Net Profit'],
            default=(None, {})
        )[0]
    else:
        best_month = None
        worst_month = None

    return {
        "Stats": stats,
        "Best Month": {"Month": best_month, **stats.get(best_month, {})} if best_month else {"Month": None},
        "Worst Month": {"Month": worst_month, **stats.get(worst_month, {})} if worst_month else {"Month": None}
    }

def calcular_weekly(df, cdi=0.12):
    df = df.copy()
    
    # Determinar colunas
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        return {}
    
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        return {}
    
    df['WeekNum'] = df[date_col].dt.isocalendar().week
    stats = {}
    weeks = sorted(df['WeekNum'].unique())
    for w in weeks:
        sub = df[df['WeekNum'] == w]
        lucro, pf, sharpe, sharpe_simp, trades = calcular_metrics(sub, cdi=cdi)
        wins = sub[sub[pnl_col] > 0]
        losses = sub[sub[pnl_col] < 0]
        win_rate = len(wins) / trades * 100 if trades > 0 else 0
        
        # Calcular média de ganho e perda
        avg_win = _safe_mean(wins[pnl_col])
        avg_loss = _safe_mean(losses[pnl_col], absolute=True)
        
        # Calcular rentabilidade total (lucro total do período)
        rentabilidade_total = lucro
        
        stats[f"Semana {w}"] = {
            "Trades": trades,
            "Net Profit": lucro,
            "Profit Factor": pf,
            "Sharpe Ratio": sharpe,
            "Sharpe DDx3": sharpe_simp,
            "Win Rate (%)": round(win_rate, 2),
            "Average Win": round(avg_win, 2),
            "Average Loss": round(avg_loss, 2),
            "Rentabilidade ($)": round(rentabilidade_total, 2)
        }
    best_week = max(stats.items(), key=lambda x: x[1]['Net Profit'])[0]
    worst_week = min((item for item in stats.items() if item[1]['Net Profit'] < 0), key=lambda x: x[1]['Net Profit'], default=(None, {}))[0]
    return {
        "Stats": stats,
        "Best Week": {"Week": best_week, **stats.get(best_week, {})},
        "Worst Week": {"Week": worst_week, **stats.get(worst_week, {})}
    }

def calcular_dados_grafico(df, capital_inicial=100000):
    """
    Calcula dados para o gráfico baseado na abertura das operações.
    PADRONIZADO: Usa apenas saldo cumulativo (sem capital inicial) para drawdown
    """
    if df.empty:
        return []
    
    df = df.copy()
    
    # Determinar colunas
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        return []
    
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        return []
    
    # Ordenar por data de abertura
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calcular equity curve trade por trade (PADRONIZADO: apenas saldo cumulativo)
    equity, peak, drawdown, _, _ = _compute_equity_components(df[pnl_col])
    df['Saldo'] = equity
    df['Saldo_Maximo'] = peak
    df['Drawdown'] = drawdown
    
    # Calcular valor da carteira (para compatibilidade, mas não usado no drawdown)
    df['Valor_Carteira'] = capital_inicial + df['Saldo']
    df['Peak_Carteira'] = capital_inicial + df['Saldo_Maximo']
    
    # PADRONIZADO: Drawdown baseado apenas no saldo cumulativo (sem capital inicial)
    df['Drawdown_Carteira'] = df['Drawdown']  # Usar o mesmo drawdown do saldo
    df['Drawdown_Percentual'] = (df['Drawdown'] / df['Saldo_Maximo'] * 100).fillna(0) if df['Saldo_Maximo'].max() != 0 else 0
    
    # Preparar dados para o gráfico
    grafico_dados = []
    
    first_date = df[date_col].iloc[0]
    grafico_dados.append({
        "date": first_date.strftime('%Y-%m-%d'),
        "fullDate": first_date.strftime('%d/%m/%Y'),
        "saldo": 0.0,  # Saldo inicial sempre 0
        "valor": float(capital_inicial),  # Patrimônio inicial
        "resultado": 0.0,  # Resultado inicial sempre 0
        "drawdown": 0.0,
        "drawdownPercent": 0.0,
        "peak": float(capital_inicial),
        "trades": 0,
        "isStart": True
    })
    
    # Dados para cada trade (incluindo trades com resultado 0)
    for i, row in df.iterrows():
        grafico_dados.append({
            "date": row[date_col].strftime('%Y-%m-%d'),
            "fullDate": row[date_col].strftime('%d/%m/%Y %H:%M'),
            "saldo": float(row['Saldo']),  # ESTE é o valor que você quer mostrar
            "valor": float(row['Valor_Carteira']),  # Patrimônio total (saldo + capital)
            "resultado": float(row['Saldo']),  # Mantém compatibilidade
            "drawdown": float(abs(row['Drawdown_Carteira'])),  # Sempre positivo
            "drawdownPercent": float(abs(row['Drawdown_Percentual'])),
            "peak": float(row['Peak_Carteira']),
            "trades": int(i + 1),
            "trade_result": float(row[pnl_col]),  # Incluir mesmo se for 0
            "trade_percent": float(row.get('operation_result_pct', 0)) if pd.notna(row.get('operation_result_pct', 0)) else 0.0,
            "month": row[date_col].strftime('%B'),
            "isStart": False
        })
    
    return grafico_dados

def calcular_dados_grafico_agrupado(df, capital_inicial=0, agrupar_por='dia'):
    """
    Calcula dados para o gráfico agrupados por período (dia, semana, mês).
    PADRONIZADO: Usa apenas saldo cumulativo (sem capital inicial) para drawdown
    """
    if df.empty:
        return []
    
    df = df.copy()
    
    # Determinar colunas
    if 'entry_date' in df.columns:
        date_col = 'entry_date'
    elif 'Abertura' in df.columns:
        date_col = 'Abertura'
    else:
        return []
    
    if 'operation_result' in df.columns:
        pnl_col = 'operation_result'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    elif 'Res. Operação' in df.columns:
        pnl_col = 'Res. Operação'
    else:
        return []
    
    # Ordenar por data de abertura
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calcular equity curve trade por trade (PADRONIZADO: apenas saldo cumulativo)
    equity, peak, drawdown, _, _ = _compute_equity_components(df[pnl_col])
    df['Saldo'] = equity
    df['Saldo_Maximo'] = peak
    df['Drawdown'] = drawdown
    
    # Calcular valor da carteira (para compatibilidade, mas não usado no drawdown)
    df['Valor_Carteira'] = capital_inicial + df['Saldo']
    df['Peak_Carteira'] = capital_inicial + df['Saldo_Maximo']
    
    # PADRONIZADO: Drawdown baseado apenas no saldo cumulativo (sem capital inicial)
    df['Drawdown_Carteira'] = df['Drawdown']  # Usar o mesmo drawdown do saldo
    df['Drawdown_Percentual'] = (df['Drawdown'] / df['Saldo_Maximo'] * 100).fillna(0) if df['Saldo_Maximo'].max() != 0 else 0
    
    if agrupar_por == 'trade':
        return calcular_dados_grafico(df, capital_inicial)
    
    # Definir agrupamento
    if agrupar_por == 'dia':
        df['Periodo'] = df[date_col].dt.date
    elif agrupar_por == 'semana':
        df['Periodo'] = df[date_col].dt.to_period('W')
    elif agrupar_por == 'mes':
        df['Periodo'] = df[date_col].dt.to_period('M')
    else:
        raise ValueError("agrupar_por deve ser 'dia', 'semana', 'mes' ou 'trade'")
    
    # Agrupar dados - CORRIGIDO para versões antigas do pandas
    try:
        # Tentar com include_groups=False (pandas >= 2.1)
        grupos = df.groupby('Periodo', include_groups=False).agg({
            pnl_col: ['sum', 'count'],
            'Saldo': 'last',
            'Saldo_Maximo': 'last',
            'Drawdown': 'min',
            'Valor_Carteira': 'last',
            'Peak_Carteira': 'last',
            'Drawdown_Carteira': 'max',
            'Drawdown_Percentual': 'max',
            date_col: 'first'
        }).reset_index()
    except TypeError:
        # Fallback para versões antigas do pandas
        grupos = df.groupby('Periodo').agg({
            pnl_col: ['sum', 'count'],
            'Saldo': 'last',
            'Saldo_Maximo': 'last',
            'Drawdown': 'min',
            'Valor_Carteira': 'last',
            'Peak_Carteira': 'last',
            'Drawdown_Carteira': 'max',
            'Drawdown_Percentual': 'max',
            date_col: 'first'
        }).reset_index()
    
    # Simplificar nomes das colunas
    grupos.columns = [
        'Periodo', 'Resultado_Periodo', 'Trades_Periodo', 
        'Saldo', 'Saldo_Maximo', 'Drawdown', 
        'Valor_Carteira', 'Peak_Carteira', 'Drawdown_Carteira',
        'Drawdown_Percentual', 'Abertura'
    ]
    
    # Preparar dados para o gráfico
    grafico_dados = []
    
    if grupos.empty:
        start_ts = pd.Timestamp.utcnow()
        return [{
            "date": start_ts.strftime('%Y-%m-%d'),
            "fullDate": start_ts.strftime('%d/%m/%Y'),
            "saldo": 0.0,
            "valor": float(capital_inicial),
            "resultado": 0.0,
            "drawdown": 0.0,
            "drawdownPercent": 0.0,
            "peak": float(capital_inicial),
            "trades": 0,
            "resultado_periodo": 0.0,
            "periodo": agrupar_por,
            "isStart": True
        }]

    primeira_data = grupos['Abertura'].iloc[0] if len(grupos) > 0 else pd.Timestamp('2024-01-01')
    grafico_dados.append({
        "date": primeira_data.strftime('%Y-%m-%d'),
        "fullDate": primeira_data.strftime('%d/%m/%Y'),
        "saldo": 0.0,  # Saldo inicial sempre 0
        "valor": capital_inicial,  # Patrimônio inicial
        "resultado": 0.0,  # Resultado inicial sempre 0
        "drawdown": 0,
        "drawdownPercent": 0,
        "peak": capital_inicial,
        "trades": 0,
        "isStart": True
    })
    
    # Dados para cada período (incluindo períodos com valores 0)
    for i, row in grupos.iterrows():
        if agrupar_por == 'dia':
            date_str = row['Periodo'].strftime('%Y-%m-%d')
            display_date = row['Periodo'].strftime('%d/%m/%Y')
        elif agrupar_por == 'semana':
            date_str = f"{row['Periodo'].year}-W{row['Periodo'].week:02d}"
            display_date = f"Semana {row['Periodo'].week}/{row['Periodo'].year}"
        elif agrupar_por == 'mes':
            date_str = f"{row['Periodo'].year}-{row['Periodo'].month:02d}"
            display_date = f"{row['Periodo'].month:02d}/{row['Periodo'].year}"
        
        # Garantir que valores 0 sejam preservados
        grafico_dados.append({
            "date": date_str,
            "fullDate": display_date,
            "saldo": float(row['Saldo']) if pd.notna(row['Saldo']) else 0.0,  # Saldo cumulativo
            "valor": float(row['Valor_Carteira']) if pd.notna(row['Valor_Carteira']) else 0.0,  # Patrimônio total
            "resultado": float(row['Saldo']) if pd.notna(row['Saldo']) else 0.0,  # Mantém compatibilidade
            "drawdown": float(abs(row['Drawdown_Carteira'])) if pd.notna(row['Drawdown_Carteira']) else 0.0,
            "drawdownPercent": float(abs(row['Drawdown_Percentual'])) if pd.notna(row['Drawdown_Percentual']) else 0.0,
            "peak": float(row['Peak_Carteira']) if pd.notna(row['Peak_Carteira']) else 0.0,
            "trades": int(row['Trades_Periodo']) if pd.notna(row['Trades_Periodo']) else 0,
            "resultado_periodo": float(row['Resultado_Periodo']) if pd.notna(row['Resultado_Periodo']) else 0.0,
            "periodo": agrupar_por,
            "isStart": False
        })
    
    return grafico_dados

def processar_backtest_completo(df, capital_inicial=100000, cdi=0.12):
    """
    Função principal que processa todos os dados do backtest
    e retorna JSON completo com todos os campos incluindo dados do gráfico.
    """
    if df.empty:
        return {
            "Performance Metrics": {},
            "Monthly Analysis": {},
            "Day of Week Analysis": {},
            "Weekly Analysis": {},
            "Equity Curve Data": {
                "trade_by_trade": [],
                "daily": [],
                "weekly": [],
                "monthly": []
            }
        }
    
    # ✅ CORREÇÃO: Otimizar cálculos com cache e verificações
    try:
        # Calcular métricas existentes com otimizações
        performance = calcular_performance(df, cdi=cdi)
        monthly = calcular_monthly(df, cdi=cdi)
        day_of_week = calcular_day_of_week(df, cdi=cdi)
        weekly = calcular_weekly(df, cdi=cdi)
        
        # ✅ CORREÇÃO: Calcular dados para gráfico com otimizações
        equity_curve_data = {
            "trade_by_trade": calcular_dados_grafico(df, capital_inicial),
            "daily": calcular_dados_grafico_agrupado(df, capital_inicial, 'dia'),
            "weekly": calcular_dados_grafico_agrupado(df, capital_inicial, 'semana'),
            "monthly": calcular_dados_grafico_agrupado(df, capital_inicial, 'mes')
        }

        # ✅ NOVO: Estatísticas complementares (posição, duração, custos)
        position_sizing = calcular_dimensionamento_posicao(df)
        trade_duration = calcular_duracao_trades_resumo(df)
        operational_costs = calcular_custos_backtest(df)
        
        # ✅ CORREÇÃO: Resposta completa com otimizações
        return {
            "Performance Metrics": performance,
            "Monthly Analysis": monthly,
            "Day of Week Analysis": day_of_week,
            "Weekly Analysis": weekly,
            "Equity Curve Data": equity_curve_data,
            "Position Sizing": position_sizing,
            "Trade Duration": trade_duration,
            "Operational Costs": operational_costs
        }
    except Exception as e:
        print(f"❌ Erro ao processar backtest completo: {e}")
        # Retornar estrutura vazia em caso de erro
        return {
            "Performance Metrics": {},
            "Monthly Analysis": {},
            "Day of Week Analysis": {},
            "Weekly Analysis": {},
            "Equity Curve Data": {
                "trade_by_trade": [],
                "daily": [],
                "weekly": [],
                "monthly": []
            }
        }

# Exemplo de uso prático
def exemplo_uso():
    """
    Exemplo de como usar o código para trazer dados trade por trade
    """
    try:
        # Carregando os dados
        df = carregar_csv('meu_arquivo.csv')
        
        # Processando backtest completo
        resultado = processar_backtest_completo(df, capital_inicial=100000, cdi=0.12)
        
        # Acessando dados trade por trade
        dados_trade_por_trade = resultado["Equity Curve Data"]["trade_by_trade"]
        
        # Exemplo de como usar os dados
        print(f"Total de trades: {len(dados_trade_por_trade) - 1}")  # -1 por causa do ponto inicial
        
        # Mostrar alguns exemplos de dados
        if len(dados_trade_por_trade) > 1:
            print("\n=== Exemplo de dados trade por trade ===")
            for i, trade in enumerate(dados_trade_por_trade[:5]):  # Primeiros 5
                print(f"Trade {i}: {trade}")
        
        # Acessar dados específicos
        if len(dados_trade_por_trade) > 1:
            ultimo_trade = dados_trade_por_trade[-1]
            print(f"\nResultado final: R$ {ultimo_trade['resultado']:.2f}")
            print(f"Valor da carteira: R$ {ultimo_trade['valor']:.2f}")
            print(f"Drawdown máximo: R$ {ultimo_trade['drawdown']:.2f}")
        
        return resultado
        
    except Exception as e:
        print(f"Erro ao processar: {e}")
        return None

# Para uso em APIs
def api_endpoint_exemplo(arquivo_csv, capital_inicial=100000):
    """
    Exemplo de como usar em uma API (Flask/FastAPI)
    """
    try:
        df = carregar_csv(arquivo_csv)
        resultado = processar_backtest_completo(df, capital_inicial=capital_inicial)
        
        # Retornar apenas os dados trade por trade se necessário
        return {
            "success": True,
            "data": resultado,
            "trade_by_trade": resultado["Equity Curve Data"]["trade_by_trade"]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }

if __name__ == "__main__":
    # Teste do exemplo
    exemplo_uso()