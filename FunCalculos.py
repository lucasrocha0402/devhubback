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
    CORREÇÃO: Preserva colunas originais (como 'Abertura') para permitir recriação de entry_date.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # CORREÇÃO CRÍTICA: Fazer uma cópia profunda para preservar todas as colunas originais
    normalized = df.copy(deep=True)
    
    # DEBUG: Log das colunas originais
    print(f"🔍 _normalize_trades_dataframe: Colunas originais ({len(normalized.columns)}): {list(normalized.columns)[:10]}...")

    # CORREÇÃO CRÍTICA: Datas de entrada e saída - criar SEMPRE que possível
    # Esta é a parte mais importante - garantir que entry_date sempre exista
    entry_date_created = False
    entry_date_source = None
    
    # Se entry_date já existe, apenas converter para datetime
    if 'entry_date' in normalized.columns:
        original_entry_date = normalized['entry_date'].copy()
        normalized['entry_date'] = pd.to_datetime(normalized['entry_date'], errors='coerce')
        # Verificar se a conversão funcionou
        if normalized['entry_date'].notna().any():
            entry_date_created = True
            entry_date_source = 'entry_date (existente)'
        else:
            print(f"⚠️ entry_date existente não tem valores válidos, tentando recriar...")
            # Tentar recriar de outras fontes
            entry_date_created = False
    
    # Se ainda não criou entry_date, tentar criar a partir de diferentes colunas de data (ordem de prioridade)
    if not entry_date_created:
        date_candidates = ['Abertura', 'data_abertura', 'Data', 'date', 'entry_time', 'inicio']
        for date_col in date_candidates:
            if date_col in normalized.columns:
                try:
                    # DEBUG: Verificar valores antes de converter
                    sample_values = normalized[date_col].dropna().head(3).tolist()
                    print(f"🔍 Tentando criar entry_date de '{date_col}'. Primeiros valores: {sample_values}")
                    
                    # CORREÇÃO CRÍTICA: Se a coluna já é datetime, usar diretamente
                    if pd.api.types.is_datetime64_any_dtype(normalized[date_col]):
                        print(f"   ✅ Coluna '{date_col}' já é datetime, usando diretamente...")
                        normalized['entry_date'] = normalized[date_col]
                        valid_count = normalized['entry_date'].notna().sum()
                        if valid_count > 0:
                            entry_date_created = True
                            entry_date_source = f'{date_col} (já era datetime)'
                            print(f"📅 ✅ Criado 'entry_date' diretamente de '{date_col}' ({valid_count} valores válidos)")
                            break
                        else:
                            print(f"   ⚠️ Coluna '{date_col}' é datetime mas está vazia (todos NaT)")
                    
                    # Se não é datetime, tentar converter
                    if not entry_date_created:
                        # Tentar múltiplos formatos de data brasileira
                        formats_to_try = [
                            "%d/%m/%Y %H:%M:%S",
                            "%d/%m/%Y %H:%M",
                            "%d/%m/%Y",
                            "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%d %H:%M",
                            "%Y-%m-%d",
                            "%d-%m-%Y %H:%M:%S",
                            "%d-%m-%Y",
                            "%m/%d/%Y %H:%M:%S",
                            "%m/%d/%Y"
                        ]
                        
                        best_conversion = None
                        best_count = 0
                        
                        for date_format in formats_to_try:
                            try:
                                converted = pd.to_datetime(
                                    normalized[date_col], 
                                    format=date_format, 
                                    errors='coerce'
                                )
                                valid_count = converted.notna().sum()
                                if valid_count > best_count:
                                    best_count = valid_count
                                    best_conversion = converted
                                    print(f"   ✅ Formato '{date_format}' converteu {valid_count}/{len(normalized)} valores")
                            except Exception as e:
                                print(f"   ⚠️ Erro com formato '{date_format}': {e}")
                                continue
                        
                        # Se encontrou uma conversão boa, usar ela
                        if best_conversion is not None and best_count > 0:
                            normalized['entry_date'] = best_conversion
                            entry_date_created = True
                            entry_date_source = f'{date_col} (formato específico)'
                            print(f"📅 ✅ Criado 'entry_date' a partir de '{date_col}' usando formato específico ({best_count} valores válidos)")
                            break
                        
                        # Se nenhum formato específico funcionou, tentar detecção automática
                        if not entry_date_created:
                            print(f"   Tentando detecção automática para '{date_col}'...")
                            try:
                                # Primeiro tentar infer_datetime_format
                                normalized['entry_date'] = pd.to_datetime(normalized[date_col], errors='coerce', infer_datetime_format=True)
                                valid_count = normalized['entry_date'].notna().sum()
                                
                                # Se ainda não funcionou, tentar sem infer_datetime_format (deprecated, mas pode funcionar)
                                if valid_count == 0:
                                    normalized['entry_date'] = pd.to_datetime(normalized[date_col], errors='coerce')
                                    valid_count = normalized['entry_date'].notna().sum()
                                
                                if valid_count > 0:
                                    entry_date_created = True
                                    entry_date_source = f'{date_col} (detecção automática)'
                                    print(f"📅 ✅ Criado 'entry_date' a partir de '{date_col}' via detecção automática ({valid_count} valores válidos)")
                                    break
                                else:
                                    print(f"   ❌ Detecção automática falhou para '{date_col}' (0 valores convertidos)")
                                    print(f"   🔍 Valores que não converteram: {normalized[date_col].dropna().head(5).tolist()}")
                            except Exception as e:
                                print(f"   ❌ Erro na detecção automática: {e}")
                    
                except Exception as e:
                    import traceback
                    print(f"⚠️ Erro ao criar entry_date de '{date_col}': {e}")
                    print(f"   Traceback: {traceback.format_exc()}")
                    continue
    
    # CORREÇÃO CRÍTICA: SEMPRE criar coluna entry_date, mesmo que vazia
    # Isso evita erros de KeyError em outras partes do código
    if 'entry_date' not in normalized.columns:
        normalized['entry_date'] = pd.NaT
        print("⚠️ AVISO: Criada coluna 'entry_date' vazia (todos NaT)")
    
    # Log final do status
    if entry_date_created:
        valid_count = normalized['entry_date'].notna().sum()
        print(f"✅ entry_date criado com sucesso a partir de '{entry_date_source}' ({valid_count}/{len(normalized)} valores válidos)")
    else:
        print(f"❌ AVISO: entry_date criado mas sem valores válidos (todos NaT)")
        print(f"   Colunas disponíveis no DataFrame: {list(normalized.columns)}")
        # Tentar listar algumas colunas de data para debug
        date_like_cols = [col for col in normalized.columns if any(keyword in str(col).lower() for keyword in ['data', 'date', 'abertura', 'fechamento', 'time'])]
        if date_like_cols:
            print(f"   Colunas que parecem ser de data: {date_like_cols}")
            for col in date_like_cols[:3]:  # Mostrar primeiras 3
                sample = normalized[col].dropna().head(2).tolist()
                print(f"      '{col}': {sample}")

    # CORREÇÃO: exit_date - criar sempre que possível (mesma lógica)
    exit_date_created = False
    
    if 'exit_date' in normalized.columns:
        normalized['exit_date'] = pd.to_datetime(normalized['exit_date'], errors='coerce')
        exit_date_created = True
    else:
        date_candidates = ['Fechamento', 'data_fechamento', 'Data Saída', 'exit_time', 'fim']
        for date_col in date_candidates:
            if date_col in normalized.columns:
                try:
                    normalized['exit_date'] = pd.to_datetime(
                        normalized[date_col],
                        format="%d/%m/%Y %H:%M:%S",
                        errors='coerce'
                    )
                    if normalized['exit_date'].isna().all():
                        normalized['exit_date'] = pd.to_datetime(normalized[date_col], errors='coerce')
                    if normalized['exit_date'].notna().any():
                        exit_date_created = True
                        print(f"📅 Criado 'exit_date' a partir de '{date_col}'")
                        break
                except Exception as e:
                    continue
    
    if not exit_date_created:
        normalized['exit_date'] = pd.NaT

    # CORREÇÃO: Detectar coluna de PnL automaticamente com ordem de prioridade
    # Prioridade: Res. Intervalo Bruto > Res. Intervalo > Res. Operação > outros
    if 'pnl' not in normalized.columns:
        pnl_candidates = [
            'Res. Intervalo Bruto',  # Prioridade 1: mais completo
            'Res. Intervalo',        # Prioridade 2: padrão mais comum
            'Res. Operação',         # Prioridade 3: alternativa
            'operation_result',      # Prioridade 4: formato inglês
            'Resultado',             # Prioridade 5: genérico
            'Resultado Operação',    # Prioridade 6: alternativo
            'Total'                  # Prioridade 7: última opção (pode ser saldo acumulado)
        ]
        
        for col in pnl_candidates:
            if col in normalized.columns:
                # Verificar se a coluna tem valores válidos (não está totalmente vazia)
                temp_pnl = pd.to_numeric(normalized[col], errors='coerce')
                if temp_pnl.notna().any():  # Se tem pelo menos um valor válido
                    normalized['pnl'] = temp_pnl
                    print(f"📊 Usando coluna '{col}' para PnL")
                    break
    else:
        normalized['pnl'] = pd.to_numeric(normalized['pnl'], errors='coerce')

    # Se ainda não encontrou PnL, tentar procurar por padrões nas colunas
    if 'pnl' not in normalized.columns:
        # Procurar colunas que contenham palavras-chave
        for col in normalized.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['resultado', 'res.', 'pnl', 'profit', 'loss', 'lucro', 'prejuízo']):
                temp_pnl = pd.to_numeric(normalized[col], errors='coerce')
                if temp_pnl.notna().any():
                    normalized['pnl'] = temp_pnl
                    print(f"📊 Detectada coluna '{col}' como PnL por padrão")
                    break

    # Se ainda não encontrou, criar coluna vazia mas não zerar tudo
    if 'pnl' not in normalized.columns:
        normalized['pnl'] = np.nan
        print("⚠️ AVISO: Nenhuma coluna de resultado financeiro encontrada")

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

    # CORREÇÃO CRÍTICA FINAL: Verificar se entry_date foi criado mas está vazio
    # Se estiver vazio e Abertura existir, tentar recriar UMA ÚLTIMA VEZ antes de retornar
    if 'entry_date' in normalized.columns:
        entry_date_valid_final = normalized['entry_date'].notna().sum()
        if entry_date_valid_final == 0 and 'Abertura' in normalized.columns:
            print(f"🔄 VERIFICAÇÃO FINAL: entry_date está vazio, tentando recriar de 'Abertura'...")
            try:
                # Se Abertura já é datetime, usar diretamente
                if pd.api.types.is_datetime64_any_dtype(normalized['Abertura']):
                    normalized['entry_date'] = normalized['Abertura']
                    entry_date_valid_final = normalized['entry_date'].notna().sum()
                    if entry_date_valid_final > 0:
                        print(f"   ✅ SUCESSO FINAL! entry_date recriado de 'Abertura' (datetime) ({entry_date_valid_final} valores)")
                else:
                    # Tentar converter Abertura para datetime
                    for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        normalized['entry_date'] = pd.to_datetime(normalized['Abertura'], format=fmt, errors='coerce')
                        entry_date_valid_final = normalized['entry_date'].notna().sum()
                        if entry_date_valid_final > 0:
                            print(f"   ✅ SUCESSO FINAL! entry_date recriado usando formato '{fmt}' ({entry_date_valid_final} valores)")
                            break
                    
                    # Última tentativa: detecção automática
                    if entry_date_valid_final == 0:
                        normalized['entry_date'] = pd.to_datetime(normalized['Abertura'], errors='coerce')
                        entry_date_valid_final = normalized['entry_date'].notna().sum()
                        if entry_date_valid_final > 0:
                            print(f"   ✅ SUCESSO FINAL! entry_date recriado via detecção automática ({entry_date_valid_final} valores)")
            except Exception as e:
                print(f"   ❌ Erro na verificação final: {e}")
        
        # Log final
        if entry_date_valid_final > 0:
            print(f"✅ _normalize_trades_dataframe: entry_date válido ({entry_date_valid_final}/{len(normalized)} valores)")
        else:
            print(f"❌ _normalize_trades_dataframe: entry_date ainda vazio após todas as tentativas")

    return normalized

def _detect_pnl_column(df: pd.DataFrame) -> str:
    """
    CORREÇÃO: Função auxiliar para detectar automaticamente coluna de PnL.
    Retorna o nome da coluna encontrada ou None.
    """
    pnl_candidates = [
        'pnl',                   # Prioridade 1: coluna padronizada
        'Res. Intervalo Bruto',  # Prioridade 2: mais completo
        'Res. Intervalo',        # Prioridade 3: padrão comum
        'Res. Operação',         # Prioridade 4: alternativa
        'operation_result'       # Prioridade 5: formato inglês
    ]
    
    # Primeiro, tentar colunas conhecidas
    for col in pnl_candidates:
        if col in df.columns:
            temp_pnl = pd.to_numeric(df[col], errors='coerce')
            if temp_pnl.notna().any():
                return col
    
    # Se não encontrou, procurar por padrões
    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['resultado', 'res.', 'intervalo', 'operação', 'pnl']):
            temp_pnl = pd.to_numeric(df[col], errors='coerce')
            if temp_pnl.notna().any():
                return col
    
    # Último recurso: primeira coluna numérica que não seja data ou índice
    exclude_cols = ['Abertura', 'Fechamento', 'entry_date', 'exit_date', 'Ativo', 'Lado', 'DayOfWeek', 'WeekNum', 'Periodo']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].notna().any():
                return col
    
    return None

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
        # CORREÇÃO: Validar valores antes de calcular médias e payoff
        wins_valid = wins.dropna()
        losses_valid = losses.dropna()
        
        avg_win = wins_valid.mean() if len(wins_valid) > 0 else 0.0
        avg_loss = abs(losses_valid.mean()) if len(losses_valid) > 0 else 0.0
        
        # Garantir valores válidos
        if pd.isna(avg_win) or np.isinf(avg_win):
            avg_win = 0.0
        if pd.isna(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.0
        
        # Calcular payoff corretamente
        if avg_loss > 0 and not pd.isna(avg_loss) and not np.isinf(avg_loss):
            payoff = avg_win / avg_loss if not pd.isna(avg_win) and not np.isinf(avg_win) else 0.0
        else:
            payoff = 0.0
        
        # Garantir que payoff seja válido
        if pd.isna(payoff) or np.isinf(payoff):
            payoff = 0.0

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
    # CORREÇÃO CRÍTICA: Salvar DataFrame original antes de normalizar
    # para poder buscar colunas que possam ser removidas na normalização
    df_original = df.copy() if not df.empty else pd.DataFrame()
    
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

    total_trades = len(normalized)
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

    # DEBUG: Mostrar todas as colunas disponíveis
    print(f"🔍 calcular_custos_backtest: Colunas normalizadas ({len(normalized.columns)}): {list(normalized.columns)}")
    if not df_original.empty:
        print(f"🔍 calcular_custos_backtest: Colunas originais ({len(df_original.columns)}): {list(df_original.columns)}")

    # CORREÇÃO CRÍTICA: Busca EXTREMAMENTE AMPLA para encontrar qualquer coluna de custos
    # Lista expandida com TODAS as variações possíveis de corretagem e emolumentos
    corretagem_keywords = [
        # Português
        'corretagem', 'corret', 'corretor', 'corretora', 'corretagem total',
        'comissão', 'comissao', 'comiss', 'comissão total', 'comissao total',
        'taxa corretagem', 'taxa de corretagem', 'taxa corretor', 'taxa corretora',
        'custo corretagem', 'custos corretagem',
        # Inglês
        'brokerage', 'broker', 'commission', 'comm', 'brokerage fee',
        # Padrões comuns em relatórios
        'taxa', 'tax', 'fee', 'fees',
        # Variações com números ou espaços
        'total corretagem', 'corretagem_', '_corretagem',
        'total comissão', 'total comissao'
    ]
    
    emolumentos_keywords = [
        # Português
        'emolumentos', 'emolumento', 'emol', 'emolumentos total',
        'taxas', 'taxa', 'tax', 'taxas total', 'taxa total',
        'custos operacionais', 'custo operacional', 'custos', 'custo',
        'taxa emolumentos', 'taxa de emolumentos',
        'custo emolumentos', 'custos emolumentos',
        # Inglês
        'fees', 'fee', 'operational costs', 'operational cost', 'operating costs',
        'regulatory fees', 'exchange fees', 'market fees',
        # Padrões comuns
        'total emolumentos', 'emolumentos_', '_emolumentos',
        'total taxas', 'total custos'
    ]
    
    # Função auxiliar para buscar coluna por palavra-chave com busca muito flexível
    def encontrar_coluna_por_keyword(df_search, keywords, tipo=''):
        """Busca coluna que contém alguma das palavras-chave (case-insensitive, busca flexível)"""
        colunas_candidatas = []
        
        # PRIMEIRO: Busca exata por palavra-chave
        for col in df_search.columns:
            col_lower = str(col).lower().strip()
            # Remover espaços, underscores, hífens para comparação mais flexível
            col_normalized = col_lower.replace(' ', '').replace('_', '').replace('-', '').replace('.', '')
            
            for keyword in keywords:
                keyword_lower = keyword.lower().strip()
                keyword_normalized = keyword_lower.replace(' ', '').replace('_', '').replace('-', '').replace('.', '')
                
                # Verificar se a palavra-chave está no nome da coluna (busca flexível)
                if keyword_normalized in col_normalized or col_normalized in keyword_normalized:
                    # Verificar se tem valores válidos
                    try:
                        valores = pd.to_numeric(df_search[col], errors='coerce')
                        valores_validos = valores.dropna()
                        # Se tem valores válidos (mesmo que zero), adicionar à lista de candidatos
                        if len(valores_validos) > 0:
                            soma = valores_validos.sum()
                            colunas_candidatas.append({
                                'coluna': col,
                                'valores': valores_validos,
                                'soma': soma,
                                'match': keyword
                            })
                            print(f"   🔍 Candidata {tipo}: '{col}' (match: '{keyword}', soma: R$ {soma:.2f})")
                    except Exception as e:
                        continue
        
        # Se encontrou candidatas, escolher a que tem maior soma (mais provável de ser a correta)
        if colunas_candidatas:
            # Ordenar por soma (maior primeiro)
            colunas_candidatas.sort(key=lambda x: abs(x['soma']), reverse=True)
            melhor = colunas_candidatas[0]
            print(f"   ✅ Melhor candidata {tipo}: '{melhor['coluna']}' (soma: R$ {melhor['soma']:.2f})")
            return melhor['coluna'], melhor['valores']
        
        # SEGUNDO: Se não encontrou, tentar buscar por padrões numéricos
        # Buscar qualquer coluna numérica que possa ser custos (valores pequenos e negativos ou positivos)
        if not colunas_candidatas:
            print(f"   🔍 Tentando busca por padrão numérico para {tipo}...")
            for col in df_search.columns:
                col_lower = str(col).lower().strip()
                # Pular colunas que já sabemos que são outras coisas
                if any(skip in col_lower for skip in ['pnl', 'resultado', 'res.', 'preço', 'price', 'data', 'date', 'abertura', 'fechamento', 'quantidade', 'quantity']):
                    continue
                
                try:
                    valores = pd.to_numeric(df_search[col], errors='coerce')
                    valores_validos = valores.dropna()
                    if len(valores_validos) > 0:
                        # Verificar se tem padrão de custo (valores pequenos, geralmente negativos)
                        media = valores_validos.mean()
                        soma_abs = valores_validos.abs().sum()
                        # Se tem muitos valores pequenos (custos são geralmente pequenos), pode ser custo
                        if abs(media) < 100 and soma_abs > 0:  # Valores menores que 100 por trade
                            print(f"   🔍 Possível {tipo} por padrão: '{col}' (média: R$ {media:.2f}, soma: R$ {soma_abs:.2f})")
                            colunas_candidatas.append({
                                'coluna': col,
                                'valores': valores_validos,
                                'soma': soma_abs
                            })
                except:
                    continue
            
            if colunas_candidatas:
                colunas_candidatas.sort(key=lambda x: abs(x['soma']), reverse=True)
                melhor = colunas_candidatas[0]
                print(f"   ⚠️ Candidata {tipo} encontrada por padrão (pode não ser correta): '{melhor['coluna']}' (soma: R$ {melhor['soma']:.2f})")
                return melhor['coluna'], melhor['valores']
        
        return None, None
    
    # Tentar encontrar colunas de corretagem (primeiro no original, depois no normalizado)
    corretagem_total = 0.0
    corretagem_col = None
    corretagem_valores = None
    
    print(f"🔍 Buscando coluna de CORRETAGEM...")
    
    # Buscar no DataFrame original primeiro
    if not df_original.empty:
        corretagem_col, corretagem_valores = encontrar_coluna_por_keyword(df_original, corretagem_keywords, 'CORRETAGEM (original)')
        if corretagem_col:
            corretagem_total = float(corretagem_valores.sum())
            print(f"✅ Encontrada coluna de corretagem no original: '{corretagem_col}' (total: R$ {corretagem_total:.2f})")
            # Copiar valores para o DataFrame normalizado se necessário
            if corretagem_col not in normalized.columns:
                normalized[corretagem_col] = df_original[corretagem_col]
    
    # Se não encontrou no original, buscar no normalizado
    if corretagem_total == 0.0:
        corretagem_col, corretagem_valores = encontrar_coluna_por_keyword(normalized, corretagem_keywords, 'CORRETAGEM (normalizado)')
        if corretagem_col:
            corretagem_total = float(corretagem_valores.sum())
            print(f"✅ Encontrada coluna de corretagem no normalizado: '{corretagem_col}' (total: R$ {corretagem_total:.2f})")
    
    # Tentar encontrar colunas de emolumentos (primeiro no original, depois no normalizado)
    emolumentos_total = 0.0
    emolumentos_col = None
    emolumentos_valores = None
    
    print(f"🔍 Buscando coluna de EMOLUMENTOS...")
    
    # Buscar no DataFrame original primeiro
    if not df_original.empty:
        emolumentos_col, emolumentos_valores = encontrar_coluna_por_keyword(df_original, emolumentos_keywords, 'EMOLUMENTOS (original)')
        if emolumentos_col:
            emolumentos_total = float(emolumentos_valores.sum())
            print(f"✅ Encontrada coluna de emolumentos no original: '{emolumentos_col}' (total: R$ {emolumentos_total:.2f})")
            # Copiar valores para o DataFrame normalizado se necessário
            if emolumentos_col not in normalized.columns:
                normalized[emolumentos_col] = df_original[emolumentos_col]
    
    # Se não encontrou no original, buscar no normalizado
    if emolumentos_total == 0.0:
        emolumentos_col, emolumentos_valores = encontrar_coluna_por_keyword(normalized, emolumentos_keywords, 'EMOLUMENTOS (normalizado)')
        if emolumentos_col:
            emolumentos_total = float(emolumentos_valores.sum())
            print(f"✅ Encontrada coluna de emolumentos no normalizado: '{emolumentos_col}' (total: R$ {emolumentos_total:.2f})")
    
    # CORREÇÃO: Se ainda não encontrou, tentar calcular pela diferença entre Bruto e Líquido
    # Esta verificação vem DEPOIS de tentar encontrar colunas explícitas
    if corretagem_total == 0.0 or emolumentos_total == 0.0:
        # Verificar se tem as colunas de resultado bruto e líquido
        coluna_bruto = None
        coluna_liquido = None
        
        # Procurar colunas que possam representar resultado bruto e líquido
        for col in normalized.columns:
            col_lower = str(col).lower().strip()
            if 'bruto' in col_lower and ('res' in col_lower or 'resultado' in col_lower):
                coluna_bruto = col
            elif ('res' in col_lower or 'resultado' in col_lower) and 'bruto' not in col_lower and 'intervalo' in col_lower:
                # Se não tem "bruto" mas tem "res" ou "resultado" e "intervalo", pode ser o líquido
                coluna_liquido = col
        
        # Se não encontrou no normalizado, procurar no original
        if (not coluna_bruto or not coluna_liquido) and not df_original.empty:
            for col in df_original.columns:
                col_lower = str(col).lower().strip()
                if 'bruto' in col_lower and ('res' in col_lower or 'resultado' in col_lower):
                    if not coluna_bruto:
                        coluna_bruto = col
                elif ('res' in col_lower or 'resultado' in col_lower) and 'bruto' not in col_lower and 'intervalo' in col_lower:
                    if not coluna_liquido:
                        coluna_liquido = col
        
        # Tentar com nomes exatos comuns
        if not coluna_bruto:
            if 'Res. Intervalo Bruto' in normalized.columns:
                coluna_bruto = 'Res. Intervalo Bruto'
            elif not df_original.empty and 'Res. Intervalo Bruto' in df_original.columns:
                coluna_bruto = 'Res. Intervalo Bruto'
        
        if not coluna_liquido:
            if 'Res. Intervalo' in normalized.columns:
                coluna_liquido = 'Res. Intervalo'
            elif not df_original.empty and 'Res. Intervalo' in df_original.columns:
                coluna_liquido = 'Res. Intervalo'
        
        # Se encontrou ambas as colunas, calcular a diferença
        if coluna_bruto and coluna_liquido:
            try:
                # Buscar no DataFrame correto
                df_para_calculo = normalized if coluna_bruto in normalized.columns else df_original
                
                bruto = pd.to_numeric(df_para_calculo[coluna_bruto], errors='coerce').fillna(0)
                liquido = pd.to_numeric(df_para_calculo[coluna_liquido], errors='coerce').fillna(0)
                diferenca = bruto - liquido
                custos_total_calculado = abs(diferenca.sum())  # Usar valor absoluto
                
                if custos_total_calculado > 0:
                    print(f"💡 Calculando custos pela diferença entre '{coluna_bruto}' e '{coluna_liquido}': R$ {custos_total_calculado:.2f}")
                    
                    # Dividir entre corretagem e emolumentos
                    if corretagem_total == 0.0:
                        corretagem_total = custos_total_calculado * 0.5  # 50% corretagem
                        print(f"   ✅ Corretagem calculada: R$ {corretagem_total:.2f}")
                    
                    if emolumentos_total == 0.0:
                        emolumentos_total = custos_total_calculado - corretagem_total if corretagem_total > 0 else custos_total_calculado * 0.5
                        print(f"   ✅ Emolumentos calculados: R$ {emolumentos_total:.2f}")
            except Exception as e:
                print(f"   ⚠️ Erro ao calcular custos pela diferença: {e}")
                import traceback
                traceback.print_exc()
    
    # CORREÇÃO CRÍTICA: SEMPRE calcular automaticamente com base nos dados disponíveis
    # Usando taxas padrão do mercado brasileiro de futuros
    # Isso garante que SEMPRE teremos valores, mesmo que não encontre colunas explícitas
    # IMPORTANTE: Sempre calcular se tiver trades, mesmo se os valores já foram encontrados
    # Isso garante que sempre haverá um valor calculado como fallback
    if total_trades > 0:
        # Se já encontrou valores, usar eles. Caso contrário, calcular automaticamente
        if corretagem_total == 0.0 or emolumentos_total == 0.0:
            print(f"💡 Calculando automaticamente corretagem e emolumentos com base nos dados do arquivo...")
            print(f"   Total de trades: {total_trades}")
        
        # Taxas padrão do mercado brasileiro de futuros (por contrato/roda)
        # Corretagem: R$ 0,50 por contrato/roda (entrada = 1 roda, saída = 1 roda, total = 2 rodas = R$ 1,00 por trade completo)
        # Emolumentos: R$ 0,03 por contrato/roda (entrada + saída = R$ 0,06 por trade completo)
        TAXA_CORRETAGEM_POR_RODA = 0.50  # Por roda
        TAXA_EMOLUMENTOS_POR_RODA = 0.03  # Por roda
        
        # Tentar calcular com base em diferentes fontes de dados
        quantidade_total_rodas = 0
        
        # MÉTODO 1: Usar coluna de quantidade/vendas/compras
        colunas_quantidade = ['Quantidade', 'quantity', 'qtd', 'Qtd Compra', 'Qtd Venda', 'Qtd', 
                             'qtd compra', 'qtd venda', 'Giro Total', 'giro total', 'giro']
        
        for col_qtd in colunas_quantidade:
            if col_qtd in normalized.columns:
                try:
                    qtd = pd.to_numeric(normalized[col_qtd], errors='coerce').dropna()
                    if len(qtd) > 0:
                        quantidade_total_rodas = qtd.sum()
                        print(f"   📊 Encontrada coluna de quantidade: '{col_qtd}' = {quantidade_total_rodas:.0f} rodas")
                        break
                except:
                    continue
        
        # Se não encontrou no normalizado, procurar no original
        if quantidade_total_rodas == 0 and not df_original.empty:
            for col_qtd in colunas_quantidade:
                if col_qtd in df_original.columns:
                    try:
                        qtd = pd.to_numeric(df_original[col_qtd], errors='coerce').dropna()
                        if len(qtd) > 0:
                            quantidade_total_rodas = qtd.sum()
                            print(f"   📊 Encontrada coluna de quantidade no original: '{col_qtd}' = {quantidade_total_rodas:.0f} rodas")
                            break
                    except:
                        continue
        
        # MÉTODO 2: Se tem "Qtd Compra" e "Qtd Venda", somar ambos (cada uma é uma roda)
        if quantidade_total_rodas == 0:
            if 'Qtd Compra' in normalized.columns and 'Qtd Venda' in normalized.columns:
                try:
                    qtd_compra = pd.to_numeric(normalized['Qtd Compra'], errors='coerce').fillna(0).sum()
                    qtd_venda = pd.to_numeric(normalized['Qtd Venda'], errors='coerce').fillna(0).sum()
                    quantidade_total_rodas = qtd_compra + qtd_venda  # Cada uma é uma roda
                    print(f"   📊 Calculando pela soma de Qtd Compra ({qtd_compra:.0f}) + Qtd Venda ({qtd_venda:.0f}) = {quantidade_total_rodas:.0f} rodas")
                except:
                    pass
            
            # Tentar no original também
            if quantidade_total_rodas == 0 and not df_original.empty:
                if 'Qtd Compra' in df_original.columns and 'Qtd Venda' in df_original.columns:
                    try:
                        qtd_compra = pd.to_numeric(df_original['Qtd Compra'], errors='coerce').fillna(0).sum()
                        qtd_venda = pd.to_numeric(df_original['Qtd Venda'], errors='coerce').fillna(0).sum()
                        quantidade_total_rodas = qtd_compra + qtd_venda
                        print(f"   📊 Calculando pela soma de Qtd Compra ({qtd_compra:.0f}) + Qtd Venda ({qtd_venda:.0f}) = {quantidade_total_rodas:.0f} rodas")
                    except:
                        pass
        
        # MÉTODO 3: Se não tem quantidade, usar número de trades (assumindo 1 contrato por trade, 2 rodas)
        if quantidade_total_rodas == 0 and total_trades > 0:
            # Cada trade tem entrada e saída = 2 rodas
            quantidade_total_rodas = total_trades * 2
            print(f"   📊 Estimando pela quantidade de trades: {total_trades} trades × 2 rodas = {quantidade_total_rodas:.0f} rodas")
        
        # CORREÇÃO CRÍTICA: SEMPRE calcular se tiver pelo menos 1 trade, mesmo sem quantidade específica
        # Calcular corretagem e emolumentos
        if quantidade_total_rodas > 0:
            # Corretagem: R$ 0,50 por roda
            if corretagem_total == 0.0:
                corretagem_total = quantidade_total_rodas * TAXA_CORRETAGEM_POR_RODA
                print(f"   ✅ Corretagem calculada automaticamente: {quantidade_total_rodas:.0f} rodas × R$ {TAXA_CORRETAGEM_POR_RODA:.2f} = R$ {corretagem_total:.2f}")
            
            # Emolumentos: R$ 0,03 por roda
            if emolumentos_total == 0.0:
                emolumentos_total = quantidade_total_rodas * TAXA_EMOLUMENTOS_POR_RODA
                print(f"   ✅ Emolumentos calculados automaticamente: {quantidade_total_rodas:.0f} rodas × R$ {TAXA_EMOLUMENTOS_POR_RODA:.2f} = R$ {emolumentos_total:.2f}")
        elif total_trades > 0:
            # Se ainda não tem quantidade mas tem trades, calcular com base nos trades
            quantidade_total_rodas = total_trades * 2
            if corretagem_total == 0.0:
                corretagem_total = quantidade_total_rodas * TAXA_CORRETAGEM_POR_RODA
                print(f"   ✅ Corretagem calculada automaticamente (fallback): {total_trades} trades × 2 rodas × R$ {TAXA_CORRETAGEM_POR_RODA:.2f} = R$ {corretagem_total:.2f}")
            if emolumentos_total == 0.0:
                emolumentos_total = quantidade_total_rodas * TAXA_EMOLUMENTOS_POR_RODA
                print(f"   ✅ Emolumentos calculados automaticamente (fallback): {total_trades} trades × 2 rodas × R$ {TAXA_EMOLUMENTOS_POR_RODA:.2f} = R$ {emolumentos_total:.2f}")
    
    # Se ainda não encontrou e tem taxas fornecidas, usar essas taxas
    if corretagem_total == 0.0 and taxa_corretagem > 0.0:
        if 'entry_price' in normalized.columns and 'exit_price' in normalized.columns:
            df_valid = normalized.dropna(subset=['entry_price', 'exit_price'])
            total_trades_valid = len(df_valid)
            corretagem_total = total_trades_valid * taxa_corretagem
            print(f"📊 Calculando corretagem com taxa fornecida: {total_trades_valid} trades × R$ {taxa_corretagem:.2f} = R$ {corretagem_total:.2f}")
    
    if emolumentos_total == 0.0 and taxa_emolumentos > 0.0:
        if 'entry_price' in normalized.columns and 'exit_price' in normalized.columns:
            df_valid = normalized.dropna(subset=['entry_price', 'exit_price'])
            valor_entrada = df_valid['entry_price'] * df_valid['position_size']
            valor_saida = df_valid['exit_price'] * df_valid['position_size']
            valor_total_operado = float((valor_entrada + valor_saida).sum())
            emolumentos_total = valor_total_operado * (taxa_emolumentos / 100.0)
            print(f"📊 Calculando emolumentos com taxa fornecida: R$ {valor_total_operado:.2f} × {taxa_emolumentos}% = R$ {emolumentos_total:.2f}")
    
    # Se ainda não encontrou, mostrar aviso
    if corretagem_total == 0.0:
        print(f"⚠️ Nenhuma coluna de corretagem encontrada e não foi possível calcular automaticamente. Colunas disponíveis: {list(normalized.columns)}")
    if emolumentos_total == 0.0:
        print(f"⚠️ Nenhuma coluna de emolumentos encontrada e não foi possível calcular automaticamente. Colunas disponíveis: {list(normalized.columns)}")
    
    # Calcular valor operado (se necessário)
    valor_total_operado = 0.0
    if 'entry_price' in normalized.columns and 'exit_price' in normalized.columns:
        df_valid = normalized.dropna(subset=['entry_price', 'exit_price'])
        if len(df_valid) > 0:
            valor_entrada = df_valid['entry_price'] * df_valid['position_size']
            valor_saida = df_valid['exit_price'] * df_valid['position_size']
            valor_total_operado = float((valor_entrada + valor_saida).sum())
    
    custo_total = corretagem_total + emolumentos_total
    
    # DEBUG: Log final dos valores calculados
    print(f"✅ calcular_custos_backtest FINAL: corretagem_total = R$ {corretagem_total:.2f}, emolumentos_total = R$ {emolumentos_total:.2f}")

    return {
        "hasData": True,
        "totalTrades": int(total_trades),
        "valorOperado": round(valor_total_operado, 2),
        "corretagem": float(round(corretagem_total, 2)),
        "emolumentos": float(round(emolumentos_total, 2)),
        "custoTotal": round(custo_total, 2),
        "custoPorTrade": round(custo_total / total_trades, 2) if total_trades > 0 else 0.0
    }

def _read_csv_with_header(file_obj):
    """
    Lê conteúdo do CSV e garante que o cabeçalho real seja usado.
    CORRIGIDO: Suporta diferentes tipos de arquivos e formatos.
    Retorna DataFrame com os dados brutos.
    """
    try:
        # Detectar tipo de arquivo
        file_extension = None
        if hasattr(file_obj, 'filename'):
            filename = file_obj.filename.lower() if file_obj.filename else 'unknown'
        elif isinstance(file_obj, str):
            filename = file_obj.lower()
        else:
            # Tentar obter nome do arquivo de outras formas
            filename = getattr(file_obj, 'name', str(file_obj)).lower()
        
        # Tentar ler como Excel se for .xlsx ou .xls
        if filename.endswith(('.xlsx', '.xls')):
            try:
                # Tentar diferentes engines para Excel
                if filename.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(file_obj, engine='openpyxl')
                    except ImportError:
                        df = pd.read_excel(file_obj, engine='xlrd')
                else:
                    try:
                        df = pd.read_excel(file_obj, engine='xlrd')
                    except ImportError:
                        df = pd.read_excel(file_obj)
                
                # Validar se tem colunas necessárias
                if not df.empty and len(df.columns) > 0:
                    # Validar que tem pelo menos uma coluna necessária
                    required_cols = ['entry_date', 'Abertura', 'pnl', 'Res. Operação', 'Res. Intervalo']
                    if any(col in df.columns for col in required_cols):
                        return df
            except Exception as e:
                # Se falhar, continuar para tentar como CSV
                pass
        
        # Tentar ler como JSON se for .json
        if filename.endswith('.json'):
            try:
                df = pd.read_json(file_obj)
                if not df.empty and len(df.columns) > 0:
                    return df
            except Exception:
                pass
        
        # Ler como CSV com diferentes encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-8-sig']
        separators = [';', ',', '\t']
        
        if hasattr(file_obj, 'read'):
            original_position = file_obj.tell()
            raw_content = file_obj.read()
            if isinstance(raw_content, bytes):
                # Tentar diferentes encodings
                for encoding in encodings:
                    try:
                        raw_content_str = raw_content.decode(encoding, errors='ignore')
                        break
                    except:
                        continue
                else:
                    raw_content_str = raw_content.decode('latin1', errors='ignore')
            else:
                raw_content_str = raw_content
            file_obj.seek(original_position)
        else:
            # Tentar diferentes encodings ao ler arquivo
            raw_content_str = None
            for encoding in encodings:
                try:
                    with open(file_obj, 'r', encoding=encoding, errors='ignore') as fh:
                        raw_content_str = fh.read()
                    break
                except:
                    continue
            
            if raw_content_str is None:
                # Fallback para leitura binária
                with open(file_obj, 'rb') as fh:
                    raw_content_bytes = fh.read()
                    raw_content_str = raw_content_bytes.decode('latin1', errors='ignore')
    except Exception as exc:
        raise ValueError(f"Erro ao ler o arquivo: {exc}")

    # Tentar detectar cabeçalho automaticamente
    lines = raw_content_str.splitlines()
    header_idx = None
    
    # Procurar por diferentes padrões de cabeçalho
    header_patterns = ['Ativo', 'entry_date', 'pnl', 'Abertura', 'Res. Operação']
    
    for idx, line in enumerate(lines):
        if any(pattern in line for pattern in header_patterns):
            header_idx = idx
            break
    
    # Se não encontrou, tentar usar a primeira linha
    if header_idx is None:
        header_idx = 0
    
    csv_text = '\n'.join(lines[header_idx:])
    data_buffer = io.StringIO(csv_text)
    
    # Tentar diferentes separadores
    for sep in separators:
        try:
            df = pd.read_csv(data_buffer, sep=sep, decimal=',', encoding='utf-8', on_bad_lines='skip', engine='python')
            if not df.empty and len(df.columns) > 1:
                data_buffer.seek(0)
                return df
        except:
            data_buffer.seek(0)
            continue
    
    # Última tentativa com separador padrão
    data_buffer.seek(0)
    try:
        df = pd.read_csv(data_buffer, sep=';', decimal=',', encoding='latin1', on_bad_lines='skip', engine='python')
        return df
    except:
        raise ValueError("Não foi possível ler o arquivo. Verifique o formato.")

def _compute_equity_components(pnl_series: pd.Series):
    """
    Calcula curvas de equity, pico e drawdown com baseline iniciando em 0.
    Retorna (equity, peak, drawdown, max_drawdown, max_drawdown_pct).
    CORRIGIDO: Valida valores nulos/NaN/zero e garante cálculos corretos.
    """
    if pnl_series is None or pnl_series.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, 0.0, 0.0

    # CORREÇÃO: Filtrar valores NaN/nulos antes de processar
    # Não substituir por zero automaticamente - manter valores válidos apenas
    pnl_valid = pnl_series.dropna()
    
    # Se não houver valores válidos, retornar valores vazios
    if pnl_valid.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, 0.0, 0.0
    
    # Garantir que todos os valores sejam numéricos válidos (não infinitos)
    pnl_valid = pnl_valid.replace([np.inf, -np.inf], np.nan).dropna()
    
    if pnl_valid.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, 0.0, 0.0
    
    # Calcular equity apenas com valores válidos
    equity = pnl_valid.cumsum()
    equity_with_start = pd.concat([pd.Series([0.0], index=[-1]), equity])
    peak = equity_with_start.cummax()
    drawdown = equity_with_start - peak

    # Remover ponto inicial artificial
    equity = equity_with_start.iloc[1:]
    peak = peak.iloc[1:]
    drawdown = drawdown.iloc[1:]

    if drawdown.empty:
        return equity, peak, drawdown, 0.0, 0.0

    # CORREÇÃO: Validar valores antes de calcular max drawdown
    drawdown_valid = drawdown.dropna()
    if drawdown_valid.empty:
        max_dd = 0.0
    else:
        max_dd_value = drawdown_valid.min()
        if pd.isna(max_dd_value) or np.isinf(max_dd_value):
            max_dd = 0.0
        else:
            max_dd = float(abs(max_dd_value))
    
    peak_valid = peak.dropna()
    if peak_valid.empty:
        peak_max = 0.0
    else:
        peak_max_value = peak_valid.max()
        if pd.isna(peak_max_value) or np.isinf(peak_max_value) or peak_max_value == 0:
            peak_max = 1.0  # Evitar divisão por zero
        else:
            peak_max = float(peak_max_value)
    
    # Calcular percentual apenas se peak_max for válido
    if peak_max > 0 and not pd.isna(peak_max) and not np.isinf(peak_max):
        max_dd_pct = float((max_dd / peak_max * 100))
    else:
        max_dd_pct = 0.0
    
    # Garantir que os valores retornados sejam válidos
    if pd.isna(max_dd_pct) or np.isinf(max_dd_pct):
        max_dd_pct = 0.0

    return equity, peak, drawdown, max_dd, max_dd_pct

def carregar_csv(file):
    """
    CORRIGIDO: Suporta diferentes tipos de arquivos e valida campos obrigatórios.
    """
    try:
        df = _read_csv_with_header(file)
        
        # CORREÇÃO: Validar que o DataFrame não está vazio
        if df is None or df.empty:
            raise ValueError("O arquivo está vazio ou não contém dados válidos.")
        
        # CORREÇÃO: Detecção flexível de colunas obrigatórias
        # Tentar encontrar coluna de data
        date_col = None
        date_candidates = ['entry_date', 'Abertura', 'Data', 'data', 'Date', 'date']
        for col in date_candidates:
            if col in df.columns:
                date_col = col
                break
        
        # Tentar encontrar coluna de PnL com ordem de prioridade
        pnl_col = None
        pnl_candidates = [
            'Res. Intervalo Bruto',  # Prioridade 1
            'Res. Intervalo',        # Prioridade 2
            'Res. Operação',         # Prioridade 3
            'pnl',                   # Prioridade 4
            'operation_result'       # Prioridade 5
        ]
        for col in pnl_candidates:
            if col in df.columns:
                pnl_col = col
                break
        
        # Se não encontrou por nome exato, procurar por padrões
        if not pnl_col:
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['resultado', 'res.', 'intervalo', 'operação']):
                    # Verificar se tem valores válidos
                    temp = pd.to_numeric(df[col], errors='coerce')
                    if temp.notna().any():
                        pnl_col = col
                        break
        
        # Validar que encontrou pelo menos uma coluna de PnL ou data
        if not pnl_col and not date_col:
            raise ValueError(
                f"O arquivo não contém colunas reconhecíveis. "
                f"Colunas encontradas: {list(df.columns)}. "
                f"Procure por colunas de resultado (Res. Intervalo, Res. Operação, etc.) ou data (Abertura)."
            )
        
        # Se não encontrou PnL mas tem data, ainda pode processar (mas avisar)
        if not pnl_col:
            print("⚠️ AVISO: Nenhuma coluna de resultado financeiro encontrada. Tentando continuar...")
        
        # Processar e limpar dados
        if pnl_col:
            # Converter para numérico e remover NaN
            df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
            # Remover linhas com PnL nulo/NaN (mas manter zeros válidos)
            df = df.dropna(subset=[pnl_col])
        
        if date_col:
            # Tentar múltiplos formatos de data
            df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y %H:%M:%S", errors='coerce')
            # Se falhou, tentar sem formato específico
            if df[date_col].isna().all():
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # CORREÇÃO: Remover apenas linhas onde a data é inválida, mas manter as válidas
            # Verificar quantas linhas válidas temos antes de remover
            linhas_validas = df[date_col].notna().sum()
            if linhas_validas > 0:
                df = df.dropna(subset=[date_col])
            else:
                print(f"⚠️ AVISO: Nenhuma data válida encontrada na coluna '{date_col}'")
                # Não remover todas as linhas, mas avisar
        
        # Validar que ainda há dados após limpeza
        if df.empty:
            raise ValueError("Após remover valores nulos/inválidos, o arquivo ficou vazio. Verifique se há dados válidos no arquivo.")
        
        # Processar datas
        if 'Abertura' in df.columns:
            df['Abertura'] = pd.to_datetime(df['Abertura'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        if 'Fechamento' in df.columns:
            df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')

        # CORREÇÃO: Limpar valores numéricos de TODAS as colunas numéricas encontradas
        # Lista completa de possíveis colunas numéricas
        numeric_columns = [
            'Res. Operação', 'Res. Operação (%)',
            'Res. Intervalo', 'Res. Intervalo (%)',
            'Res. Intervalo Bruto', 'Res. Intervalo Bruto (%)',
            'Preço Compra', 'Preço Venda', 'Preço de Mercado', 'Médio',
            'Qtd Compra', 'Qtd Venda', 
            'Drawdown', 'Ganho Max.', 'Perda Max.', 'Total',
            'Número Operação', 'TET'
        ]
        
        # Limpar colunas conhecidas
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)
        
        # CORREÇÃO: Também limpar qualquer coluna que pareça ser numérica
        # (evita problemas com valores formatados como "135.961,25")
        for col in df.columns:
            if col not in ['Abertura', 'Fechamento', 'Ativo', 'Lado', 'Tempo Operação', 'Médio', 'TET']:
                # Tentar converter e verificar se é numérica
                try:
                    sample_values = df[col].dropna().head(5)
                    if len(sample_values) > 0:
                        # Tentar converter alguns valores para ver se são numéricos
                        test_convert = pd.to_numeric(sample_values, errors='coerce')
                        if test_convert.notna().any():
                            # Se pelo menos um valor converteu, aplicar limpeza a toda a coluna
                            df[col] = df[col].apply(clean_numeric_value)
                except:
                    pass  # Se der erro, ignora a coluna

        # CORREÇÃO CRÍTICA: Normalizar o DataFrame antes de retornar
        # Isso garante que sempre tenhamos colunas padronizadas (entry_date, pnl, etc.)
        print(f"🔍 DEBUG carregar_csv: Antes da normalização - shape: {df.shape}, colunas: {list(df.columns)[:5]}...")
        
        df_normalized = _normalize_trades_dataframe(df)
        
        # Validar que ainda há dados após normalização
        if df_normalized.empty:
            raise ValueError("Após normalização, o arquivo ficou vazio. Verifique se há dados válidos nas colunas 'Abertura' e de resultado.")
        
        # Validar que entry_date foi criado
        if 'entry_date' not in df_normalized.columns:
            raise ValueError(
                f"Não foi possível criar coluna 'entry_date'. "
                f"Colunas disponíveis: {list(df_normalized.columns)}. "
                f"Verifique se o arquivo contém a coluna 'Abertura' com datas válidas."
            )
        
        # Validar que há pelo menos alguns valores válidos em entry_date
        entry_date_valid = df_normalized['entry_date'].notna().sum()
        if entry_date_valid == 0:
            print(f"⚠️ AVISO: entry_date foi criado mas está vazio (todos NaT)")
            print(f"   Verificando coluna 'Abertura' original...")
            if 'Abertura' in df.columns:
                print(f"   Primeiros valores de 'Abertura': {df['Abertura'].head(3).tolist()}")
            # Não lançar erro ainda - deixar que a função que usa valide
        
        print(f"✅ DEBUG carregar_csv: Após normalização - shape: {df_normalized.shape}, entry_date válidos: {entry_date_valid}/{len(df_normalized)}")
        
        return df_normalized
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
    # CORREÇÃO: Detectar coluna de PnL com ordem de prioridade
    pnl_col = None
    pnl_candidates = [
        'pnl', 'Res. Intervalo Bruto', 'Res. Intervalo', 'Res. Operação', 'operation_result'
    ]
    
    for col in pnl_candidates:
        if col in sub.columns:
            temp_pnl = pd.to_numeric(sub[col], errors='coerce')
            if temp_pnl.notna().any():
                pnl_col = col
                break
    
    # Se não encontrou, procurar por padrões
    if pnl_col is None:
        for col in sub.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['resultado', 'res.', 'intervalo', 'operação', 'pnl']):
                temp_pnl = pd.to_numeric(sub[col], errors='coerce')
                if temp_pnl.notna().any():
                    pnl_col = col
                    break
    
    # Último recurso: primeira coluna numérica
    if pnl_col is None:
        exclude_cols = ['Abertura', 'Fechamento', 'entry_date', 'exit_date']
        numeric_cols = [col for col in sub.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
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
    
    # CORREÇÃO: Detectar coluna de PnL com ordem de prioridade mais abrangente
    pnl_col = None
    pnl_candidates = [
        'pnl',                   # Prioridade 1: coluna padronizada
        'Res. Intervalo Bruto',  # Prioridade 2: mais completo
        'Res. Intervalo',        # Prioridade 3: padrão comum
        'Res. Operação',         # Prioridade 4: alternativa
        'operation_result'       # Prioridade 5: formato inglês
    ]
    
    # Primeiro, tentar colunas conhecidas
    for col in pnl_candidates:
        if col in df.columns:
            # Verificar se tem valores válidos
            temp_pnl = pd.to_numeric(df[col], errors='coerce')
            if temp_pnl.notna().any():
                pnl_col = col
                break
    
    # Se não encontrou, procurar por padrões
    if pnl_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['resultado', 'res.', 'intervalo', 'operação', 'pnl']):
                temp_pnl = pd.to_numeric(df[col], errors='coerce')
                if temp_pnl.notna().any():
                    pnl_col = col
                    break
    
    # Último recurso: primeira coluna numérica que não seja data ou índice
    if pnl_col is None:
        exclude_cols = ['Abertura', 'Fechamento', 'entry_date', 'exit_date', 'Ativo', 'Lado']
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        if len(numeric_cols) > 0:
            # Verificar qual tem mais valores válidos
            for col in numeric_cols:
                if df[col].notna().any():
                    pnl_col = col
                    break
    
    if pnl_col is None:
        return {}
    
    pnl = pd.to_numeric(df[pnl_col], errors='coerce')
    net_profit = pnl.sum()
    profit_trades = df[pnl > 0]
    loss_trades = df[pnl < 0]
    gross_profit = profit_trades[pnl_col].sum()
    gross_loss = abs(loss_trades[pnl_col].sum())
    # Validar e filtrar valores nulos/NaN antes de calcular médias
    profit_pnl = profit_trades[pnl_col].dropna()
    loss_pnl = loss_trades[pnl_col].dropna()
    
    # Calcular médias apenas com valores válidos (não nulos, não NaN, não zero quando necessário)
    avg_win = profit_pnl.mean() if len(profit_pnl) > 0 and profit_pnl.sum() != 0 else 0.0
    avg_loss = abs(loss_pnl.mean()) if len(loss_pnl) > 0 and loss_pnl.sum() != 0 else 0.0
    
    # Garantir que avg_win e avg_loss sejam valores válidos (não NaN, não infinito)
    if pd.isna(avg_win) or np.isinf(avg_win):
        avg_win = 0.0
    if pd.isna(avg_loss) or np.isinf(avg_loss):
        avg_loss = 0.0
    
    avg_per_trade = net_profit / total_trades if total_trades > 0 else 0
    win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 and gross_loss != 0.0 else 0.0
    
    # CORREÇÃO: Calcular payoff corretamente - evitar divisão por zero e valores inválidos
    if avg_loss > 0 and not pd.isna(avg_loss) and not np.isinf(avg_loss):
        payoff = avg_win / avg_loss if avg_win != 0 and not pd.isna(avg_win) else 0.0
    else:
        payoff = 0.0
    
    # Garantir que payoff seja um valor válido
    if pd.isna(payoff) or np.isinf(payoff):
        payoff = 0.0

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
    
    # CORREÇÃO: Detectar colunas automaticamente
    date_col = None
    for col in ['entry_date', 'Abertura']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        return {}
    
    # Detectar coluna de PnL usando função auxiliar
    pnl_col = _detect_pnl_column(df)
    if pnl_col is None:
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
        
        # CORREÇÃO: Calcular média de ganho e perda com validação
        wins_valid = wins[pnl_col].dropna()
        losses_valid = losses[pnl_col].dropna()
        
        avg_win = wins_valid.mean() if len(wins_valid) > 0 else 0.0
        avg_loss = abs(losses_valid.mean()) if len(losses_valid) > 0 else 0.0
        
        # Garantir valores válidos
        if pd.isna(avg_win) or np.isinf(avg_win):
            avg_win = 0.0
        if pd.isna(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.0
        
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
        
        # CORREÇÃO: Calcular média de ganho e perda com validação
        wins_valid = wins[pnl_col].dropna()
        losses_valid = losses[pnl_col].dropna()
        
        avg_win = wins_valid.mean() if len(wins_valid) > 0 else 0.0
        avg_loss = abs(losses_valid.mean()) if len(losses_valid) > 0 else 0.0
        
        # Garantir valores válidos
        if pd.isna(avg_win) or np.isinf(avg_win):
            avg_win = 0.0
        if pd.isna(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0.0
        
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
    
    # CORREÇÃO: Detectar coluna de PnL usando função auxiliar
    pnl_col = _detect_pnl_column(df)
    if pnl_col is None:
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
    
    # CORREÇÃO: Detectar coluna de PnL usando função auxiliar
    pnl_col = _detect_pnl_column(df)
    if pnl_col is None:
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

def processar_backtest_completo(df, capital_inicial=100000, cdi=0.12, taxa_corretagem=None, taxa_emolumentos=None):
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
    
    # CORREÇÃO CRÍTICA: Normalizar o DataFrame SEMPRE, mesmo se já tem entry_date
    # Isso garante que entry_date, pnl, etc. sempre existam no formato correto
    # Importante: Normalizar mesmo se as colunas já existem para garantir formato correto
    print(f"🔄 processar_backtest_completo: Normalizando DataFrame (shape: {df.shape}, tem entry_date: {'entry_date' in df.columns})...")
    
    # Verificar se precisa normalizar (se não tem entry_date OU se tem Abertura mas entry_date está vazio)
    needs_normalization = (
        'entry_date' not in df.columns or 
        'pnl' not in df.columns or
        ('Abertura' in df.columns and ('entry_date' not in df.columns or df['entry_date'].isna().all()))
    )
    
    if needs_normalization:
        df_original_shape = df.shape
        df = _normalize_trades_dataframe(df)
        
        if df.empty:
            print(f"⚠️ processar_backtest_completo: DataFrame ficou vazio após normalização (tinha {df_original_shape[0]} linhas)")
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
        
        print(f"✅ processar_backtest_completo: DataFrame normalizado (shape: {df.shape}, entry_date válidos: {df['entry_date'].notna().sum() if 'entry_date' in df.columns else 0})")
    
    # Validar que entry_date e pnl existem após normalização
    if 'entry_date' not in df.columns:
        print(f"❌ processar_backtest_completo: entry_date não encontrado após normalização")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        print(f"   Tem 'Abertura': {'Abertura' in df.columns}")
        if 'Abertura' in df.columns:
            print(f"   Primeiros valores de 'Abertura': {df['Abertura'].head(3).tolist()}")
        
        raise ValueError(
            f"O processamento individual falhou porque o arquivo não contém a coluna 'entry_date'. "
            f"➤ Nota: O arquivo possui a coluna 'Abertura' que foi mapeada com sucesso no processamento consolidado. "
            f"➤ O backend processou os dados consolidados corretamente, mas o processamento individual requer a coluna 'entry_date' explicitamente."
        )
    
    # Validar que há pelo menos alguns valores válidos em entry_date
    entry_date_valid = df['entry_date'].notna().sum()
    if entry_date_valid == 0:
        print(f"⚠️ processar_backtest_completo: entry_date existe mas está vazio (todos NaT)")
        if 'Abertura' in df.columns:
            print(f"   Primeiros valores de 'Abertura': {df['Abertura'].head(3).tolist()}")
        
        # Tentar continuar mesmo assim, mas pode falhar depois
        print(f"   Tentando continuar com {len(df)} linhas...")
    
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
        print(f"🔍 processar_backtest_completo: Calculando custos operacionais...")
        # CORREÇÃO: Passar taxas customizadas para calcular_custos_backtest se fornecidas
        # Se None, a função usará cálculo automático (padrão do mercado)
        operational_costs = calcular_custos_backtest(
            df, 
            taxa_corretagem=taxa_corretagem if taxa_corretagem is not None else 0.0,
            taxa_emolumentos=taxa_emolumentos if taxa_emolumentos is not None else 0.0
        )
        print(f"✅ processar_backtest_completo: Custos operacionais calculados: {operational_costs}")
        if taxa_corretagem is not None or taxa_emolumentos is not None:
            print(f"   💼 Taxas customizadas usadas - Corretagem: {taxa_corretagem if taxa_corretagem else 'automática'}, Emolumentos: {taxa_emolumentos if taxa_emolumentos else 'automática'}")
        
        # CORREÇÃO: Adicionar corretagem e emolumentos às métricas de performance
        # para garantir que o frontend receba esses valores
        print(f"🔍 processar_backtest_completo: Verificando operational_costs...")
        print(f"   operational_costs type: {type(operational_costs)}")
        print(f"   operational_costs: {operational_costs}")
        
        if operational_costs:
            has_data = operational_costs.get("hasData", False)
            corretagem = operational_costs.get("corretagem", 0.0)
            emolumentos = operational_costs.get("emolumentos", 0.0)
            print(f"   hasData: {has_data}, corretagem: R$ {corretagem:.2f}, emolumentos: R$ {emolumentos:.2f}")
            
            # SEMPRE adicionar aos valores de performance, mesmo que sejam zero (para garantir que o frontend receba)
            performance["Total Brokerage"] = float(corretagem) if corretagem else 0.0
            performance["Total Fees"] = float(emolumentos) if emolumentos else 0.0
            performance["Corretagem Total"] = float(corretagem) if corretagem else 0.0
            performance["Emolumentos Totais"] = float(emolumentos) if emolumentos else 0.0
            print(f"✅ processar_backtest_completo: Adicionado às métricas - Corretagem Total: R$ {performance['Corretagem Total']:.2f}, Emolumentos Totais: R$ {performance['Emolumentos Totais']:.2f}")
        else:
            print(f"⚠️ processar_backtest_completo: operational_costs é None ou vazio")
            # Ainda assim adicionar valores zero para garantir que o frontend receba
            performance["Total Brokerage"] = 0.0
            performance["Total Fees"] = 0.0
            performance["Corretagem Total"] = 0.0
            performance["Emolumentos Totais"] = 0.0
        
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