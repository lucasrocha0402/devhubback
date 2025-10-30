from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import openai
from openai import OpenAI as _OpenAIClient
from FunMultiCalculos import processar_multiplos_arquivos, processar_multiplos_arquivos_comparativo
from Correlacao import *
from FunCalculos import carregar_csv, calcular_performance, calcular_day_of_week, calcular_monthly, processar_backtest_completo, calcular_dados_grafico
import dotenv
import os.path as _path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Carregar variáveis de ambiente de múltiplas localizações para maior robustez
# 1) .env do diretório atual (python-freela/.env)
dotenv.load_dotenv()
# 2) .env explícito neste diretório
dotenv.load_dotenv(dotenv_path=_path.join(_path.dirname(__file__), '.env'))
# 3) .env do frontend (project/.env), caso a chave tenha sido colocada lá por engano
dotenv.load_dotenv(dotenv_path=_path.join(_path.dirname(__file__), '..', 'project', '.env'))

# main.py
app = Flask(__name__)

# Configuração CORS para permitir acesso do frontend
CORS(app, origins=[
    'http://localhost:4173',  # Desenvolvimento local
    'http://localhost:5173',  # Vite dev server
    'http://localhost:3000',  # Desenvolvimento local (alternativo)
    'https://devhubtrader.com.br',  # Produção
    'https://www.devhubtrader.com.br',  # Produção com www
    'http://devhubtrader.com.br',  # Produção sem SSL
    'http://www.devhubtrader.com.br'  # Produção sem SSL com www
], supports_credentials=True, allow_headers=['Content-Type', 'Authorization', 'x-openai-key'], methods=['GET','POST','OPTIONS'])

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Custom JSON provider para lidar com tipos numpy (Flask 2.3+)
from flask.json.provider import JSONProvider
import json

class NumpyJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(self._convert_numpy_types(obj), **kwargs)
    
    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)
    
    def _convert_numpy_types(self, obj):
        """Converte tipos numpy para tipos Python nativos"""
        if isinstance(obj, dict):
            return {str(k): self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.bytes_):
            return obj.decode('utf-8')
        elif pd.isna(obj) or obj is None:
            return None
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (pd.Period, pd.Timestamp)):
            return str(obj)
        else:
            return obj

# Configurar o provider customizado
app.json_provider_class = NumpyJSONProvider

# Configuração da chave da API do OpenAI (compat com SDK novo)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("[WARN] OPENAI_API_KEY não encontrado nas variáveis de ambiente. Rotas que usam OpenAI irão falhar até que seja configurado.")

# ============ MIDDLEWARE PARA LOG ============
@app.before_request
def log_request_info():
    """Log das requisições para debug"""
    # Silent request logging
    pass

# ============ ROTA RAIZ ============
@app.route('/', methods=['GET'])
def root():
    """Rota raiz para verificar se o servidor está funcionando"""
    return jsonify({
        "status": "online",
        "message": "DevHub Trader Backend API",
        "version": "1.0.0",
        "endpoints": [
            "/api/tabela",
            "/api/tabela-multipla", 
            "/api/equity-curve",
            "/api/backtest-completo",
            "/api/correlacao",
            "/api/disciplina-completa",
            "/api/trades",
            "/api/trades/summary",
            "/api/trades/daily-metrics",
            "/api/trades/metrics-from-data",
            "/chat"
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Rota de health check para monitoramento"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "devhub-backend",
        "openai_key_detected": bool(os.getenv("OPENAI_API_KEY"))
    })

@app.route('/api/test-metrics', methods=['POST'])
def test_metrics():
    """Endpoint de teste para verificar se a API de métricas está funcionando"""
    try:
        # Dados de teste simples
        test_data = {
            'trades': [
                {
                    'entry_date': '2024-01-01T10:00:00',
                    'exit_date': '2024-01-01T10:30:00',
                    'pnl': 100
                },
                {
                    'entry_date': '2024-01-01T11:00:00',
                    'exit_date': '2024-01-01T11:15:00',
                    'pnl': -50
                }
            ],
            'capital_inicial': 100000,
            'cdi': 0.12
        }
        
        # Simular o processamento
        df = pd.DataFrame(test_data['trades'])
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        
        # Testar import do FunCalculos
        try:
            from FunCalculos import processar_backtest_completo
            resultado = processar_backtest_completo(df, capital_inicial=100000, cdi=0.12)
            
            return jsonify({
                "status": "success",
                "message": "API de métricas funcionando corretamente",
                "test_trades": len(df),
                "performance_metrics": resultado.get("Performance Metrics", {})
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Erro no FunCalculos: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Erro no teste: {str(e)}"
        }), 500

# ============ FUNÇÃO AUXILIAR PARA ENCODING ============

def clean_numeric_value(value):
    """Converte valores numéricos brasileiros para float"""
    if pd.isna(value) or value == '':
        return np.nan
    
    # Converter para string se não for
    str_value = str(value)
    
    # Remover espaços em branco
    str_value = str_value.strip()
    
    # Se já for um número, retornar
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remover pontos (separador de milhares) e trocar vírgula por ponto
    # Exemplo: "371.520,00" -> "371520.00"
    if ',' in str_value:
        # Separar parte inteira da decimal
        parts = str_value.split(',')
        if len(parts) == 2:
            integer_part = parts[0].replace('.', '')  # Remove pontos da parte inteira
            decimal_part = parts[1]
            cleaned_value = f"{integer_part}.{decimal_part}"
        else:
            cleaned_value = str_value.replace('.', '').replace(',', '.')
    else:
        # Se não tem vírgula, pode ser que tenha apenas pontos como separadores de milhares
        # ou seja um número sem decimais
        if str_value.count('.') > 1:
            # Múltiplos pontos = separadores de milhares
            cleaned_value = str_value.replace('.', '')
        else:
            cleaned_value = str_value
    
    try:
        return float(cleaned_value)
    except ValueError:
        return np.nan

def carregar_csv_trades(file_path_or_file):
    """Carrega CSV da planilha de trades com mapeamento específico e parsing melhorado"""
    try:
        if hasattr(file_path_or_file, 'read'):
            # É um arquivo upload - usar mesmos parâmetros da função original
            df = pd.read_csv(file_path_or_file, skiprows=5, sep=';', encoding='latin1', decimal=',')
        else:
            # É um caminho de arquivo
            df = pd.read_csv(file_path_or_file, skiprows=5, sep=';', encoding='latin1', decimal=',')
        
        # Processar datas conforme função original - com verificação de colunas
        if 'Abertura' in df.columns:
            df['Abertura']   = pd.to_datetime(df['Abertura'],   format="%d/%m/%Y %H:%M:%S", errors='coerce')
        if 'Fechamento' in df.columns:
            df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        
        # Usar função de limpeza para valores numéricos
        numeric_columns = ['Res. Operação', 'Res. Operação (%)', 'Preço Compra', 'Preço Venda', 
                          'Preço de Mercado', 'Médio', 'Res. Intervalo', 'Res. Intervalo (%)',
                          'Res. Intervalo Bruto', 'Res. Intervalo Bruto (%)',
                          'Drawdown', 'Ganho Max.', 'Perda Max.', 'Qtd Compra', 'Qtd Venda']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)
        
        # Renomear colunas para padronizar
        column_mapping = {
            'Ativo': 'symbol',
            'Abertura': 'entry_date',
            'Fechamento': 'exit_date',
            'Tempo Operação': 'duration_str',
            'Qtd Compra': 'qty_buy',
            'Qtd Venda': 'qty_sell',
            'Lado': 'direction',
            'Preço Compra': 'entry_price',
            'Preço Venda': 'exit_price',
            'Preço de Mercado': 'market_price',
            'Médio': 'avg_price',
            # Algumas planilhas usam "Res. Intervalo Bruto"
            'Res. Intervalo': 'pnl',
            'Res. Intervalo (%)': 'pnl_pct',
            'Res. Intervalo Bruto': 'pnl',
            'Res. Intervalo Bruto (%)': 'pnl_pct',
            'Número Operação': 'trade_number',
            'Res. Operação': 'operation_result',
            'Res. Operação (%)': 'operation_result_pct',
            'Drawdown': 'drawdown',
            'Ganho Max.': 'max_gain',
            'Perda Max.': 'max_loss',
            'TET': 'tet',
            'Total': 'total'
        }
        
        # Renomear colunas existentes
        df = df.rename(columns=column_mapping)
        
        # Converter direção para formato padrão
        if 'direction' in df.columns:
            df['direction'] = df['direction'].map({'C': 'long', 'V': 'short'}).fillna('long')
        
        # Usar os resultados já processados (agora com valores limpos)
        if 'operation_result' in df.columns:
            df['pnl'] = df['operation_result']
        if 'operation_result_pct' in df.columns:
            df['pnl_pct'] = df['operation_result_pct']
        
        # Calcular duração em horas se não existir
        if 'entry_date' in df.columns and 'exit_date' in df.columns:
            if df['entry_date'].notna().any() and df['exit_date'].notna().any():
                df['duration_hours'] = (df['exit_date'] - df['entry_date']).dt.total_seconds() / 3600
        
        return df
        
    except Exception as e:
        raise ValueError(f"Erro ao processar CSV: {e}")

# Função carregar_csv_safe melhorada com encoding robusto
def carregar_csv_safe(file_path_or_file):
    """Função auxiliar para carregar CSV com encoding seguro baseada na função original"""
    try:
        # Tentar diferentes encodings e formatos
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        formats_to_try = [
            {'skiprows': 0, 'sep': ',', 'encoding': None},
            {'skiprows': 5, 'sep': ';', 'encoding': None, 'decimal': ','},
            {'skiprows': 0, 'sep': ',', 'encoding': None},
            {'skiprows': 5, 'sep': ';', 'encoding': None, 'decimal': ','}
        ]
        
        df = None
        last_error = None
        
        for encoding in encodings_to_try:
            for format_config in formats_to_try:
                try:
                    if hasattr(file_path_or_file, 'read'):
                        file_path_or_file.seek(0)  # Reset file pointer
                        format_config['encoding'] = encoding
                        df = pd.read_csv(file_path_or_file, **format_config)
                    else:
                        format_config['encoding'] = encoding
                        df = pd.read_csv(file_path_or_file, **format_config)
                    
                    # Verificar se tem colunas esperadas
                    expected_columns = ['entry_date', 'exit_date', 'pnl', 'Abertura', 'Fechamento', 'Res. Operação', 'Res. Intervalo']
                    found_columns = [col for col in expected_columns if col in df.columns]
                    
                    if found_columns:
                        break
                    else:
                        continue
                        
                except Exception as e:
                    last_error = e
                    continue
            
            if df is not None and len(df.columns) > 0:
                break
        
        if df is None or len(df.columns) == 0:
            raise ValueError(f"Não foi possível ler o CSV com nenhum encoding/formato. Último erro: {last_error}")
        
        # Não criar colunas duplicadas aqui - vamos renomear diretamente
        
        # Processar datas conforme função original - com verificação de colunas
        if 'Abertura' in df.columns:
            df['Abertura']   = pd.to_datetime(df['Abertura'],   format="%d/%m/%Y %H:%M:%S", errors='coerce')
        if 'Fechamento' in df.columns:
            df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
        
        # Usar função de limpeza para valores numéricos
        numeric_columns = ['Res. Operação', 'Res. Operação (%)', 'Preço Compra', 'Preço Venda', 
                          'Preço de Mercado', 'Médio', 'Res. Intervalo', 'Res. Intervalo (%)',
                          'Drawdown', 'Ganho Max.', 'Perda Max.', 'Qtd Compra', 'Qtd Venda']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric_value)
        
        # Renomear colunas para padronizar
        column_mapping = {
            'Ativo': 'symbol',
            'Abertura': 'entry_date',
            'Fechamento': 'exit_date',
            'Tempo Operação': 'duration_str',
            'Qtd Compra': 'qty_buy',
            'Qtd Venda': 'qty_sell',
            'Lado': 'direction',
            'Preço Compra': 'entry_price',
            'Preço Venda': 'exit_price',
            'Preço de Mercado': 'market_price',
            'Médio': 'avg_price',
            'Res. Intervalo': 'pnl',
            'Res. Intervalo (%)': 'pnl_pct',
            'Res. Intervalo Bruto': 'pnl',
            'Res. Intervalo Bruto (%)': 'pnl_pct',
            'Número Operação': 'trade_number',
            'Res. Operação': 'operation_result',
            'Res. Operação (%)': 'operation_result_pct',
            'Drawdown': 'drawdown',
            'Ganho Max.': 'max_gain',
            'Perda Max.': 'max_loss',
            'TET': 'tet',
            'Total': 'total'
        }
        
        # Renomear colunas existentes
        df = df.rename(columns=column_mapping)
        
        # Converter direção para formato padrão
        if 'direction' in df.columns:
            df['direction'] = df['direction'].map({'C': 'long', 'V': 'short'}).fillna('long')
        
        # Garantir que a coluna 'pnl' exista e seja numérica
        if 'pnl' not in df.columns and 'operation_result' in df.columns:
            df['pnl'] = df['operation_result']
        if 'pnl' in df.columns:
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        
        # Calcular duração em horas se não existir
        if 'entry_date' in df.columns and 'exit_date' in df.columns:
            # Garantir que as datas são datetime
            try:
                if hasattr(df['entry_date'], 'dtype') and df['entry_date'].dtype == 'object':
                    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
                if hasattr(df['exit_date'], 'dtype') and df['exit_date'].dtype == 'object':
                    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
                
                # Calcular duração apenas se as datas são válidas
                valid_dates = df['entry_date'].notna() & df['exit_date'].notna()
                if valid_dates.any():
                    try:
                        # Calcular duração corretamente usando Series
                        duration_series = (df.loc[valid_dates, 'exit_date'] - df.loc[valid_dates, 'entry_date'])
                        df.loc[valid_dates, 'duration_hours'] = duration_series.dt.total_seconds() / 3600
                    except Exception as e:
                        print(f"🔍 DEBUG: Erro ao calcular duração: {e}")
                        # Se houver erro, não calcular duração
                        pass
            except Exception as e:
                print(f"🔍 DEBUG: Erro ao processar datas: {e}")
                # Se houver erro, tentar converter de forma mais simples
                try:
                    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
                    df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
                except:
                    pass
        
        print(f"🔍 DEBUG: DataFrame final, shape: {df.shape}")
        print(f"🔍 DEBUG: Colunas finais: {df.columns.tolist()}")
        return df
                
    except Exception as e:
        print(f"🔍 DEBUG: Erro em carregar_csv_safe: {e}")
        raise ValueError(f"Erro ao processar CSV: {e}")

def processar_trades(df: pd.DataFrame, arquivo_para_indices: Dict[int, str] = None) -> List[Dict]:
    """Converte DataFrame em lista de trades para o frontend
    - Inclui também operações em aberto (sem exit_date), usando entry_date como fallback para exit_date
    - Mantém PnL informado no CSV
    """
    trades = []

    print(f"🔍 Processando trades - DataFrame shape: {df.shape}")
    print(f"📅 Colunas disponíveis: {list(df.columns)}")

    # Verificar se a coluna mínima necessária existe
    required_columns = ['entry_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return trades

    processed_count = 0
    skipped_count = 0

    for idx, row in df.iterrows():
        # Validar entry_date
        entry_date = row.get('entry_date')
        if pd.isna(entry_date):
            skipped_count += 1
            continue

        # exit_date pode ser ausente em operações abertas; usar entry_date como fallback
        raw_exit_date = row.get('exit_date')
        is_open = pd.isna(raw_exit_date)
        exit_date = raw_exit_date if pd.notna(raw_exit_date) else entry_date

        # Determinar a estratégia baseada no arquivo de origem (se disponível)
        strategy = "Manual"
        if arquivo_para_indices and idx in arquivo_para_indices:
            filename = arquivo_para_indices[idx]
            strategy = filename.replace('.csv', '').replace('.CSV', '')

        trade = {
            "entry_date": entry_date.isoformat() if pd.notna(entry_date) else None,
            "exit_date": exit_date.isoformat() if pd.notna(exit_date) else None,
            "entry_price": float(row.get('entry_price', 0)) if pd.notna(row.get('entry_price')) else 0,
            "exit_price": float(row.get('exit_price', 0)) if pd.notna(row.get('exit_price')) else 0,
            "pnl": float(row.get('pnl', 0)) if pd.notna(row.get('pnl')) else 0,
            "pnl_pct": float(row.get('pnl_pct', 0)) if pd.notna(row.get('pnl_pct')) else 0,
            "direction": row.get('direction', 'long'),
            "symbol": str(row.get('symbol', 'N/A')),
            "strategy": strategy,
            "quantity_total": int(row.get('qty_buy')) + int(row.get('qty_sell')) if pd.notna(row.get('qty_buy')) and pd.notna(row.get('qty_sell')) else 0,
            "quantity_compra": int(row.get('qty_buy', 0)) if pd.notna(row.get('qty_buy')) else 0,
            "quantity_venda": int(row.get('qty_sell', 0)) if pd.notna(row.get('qty_sell')) else 0,
            "duration": float(row.get('duration_hours', 0)) if pd.notna(row.get('duration_hours')) else 0,
            "drawdown": float(row.get('drawdown', 0)) if pd.notna(row.get('drawdown')) else 0,
            "max_gain": float(row.get('max_gain', 0)) if pd.notna(row.get('max_gain')) else 0,
            "max_loss": float(row.get('max_loss', 0)) if pd.notna(row.get('max_loss')) else 0,
            "is_open": bool(is_open)
        }
        trades.append(trade)
        processed_count += 1

    print(f"✅ Trades processados: {processed_count}, pulados: {skipped_count}")
    return trades

def calcular_estatisticas_temporais(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estatísticas temporais com serialização JSON correta"""
    if df.empty or 'entry_date' not in df.columns:
        return {}
    
    df_valid = df.dropna(subset=['entry_date', 'pnl'])
    
    if df_valid.empty:
        return {}
    
    # Por dia da semana
    df_valid['day_of_week'] = df_valid['entry_date'].dt.day_name()
    day_stats = df_valid.groupby('day_of_week')['pnl'].agg(['count', 'sum', 'mean']).round(2)
    
    # Por mês - converter Period para string
    df_valid['month'] = df_valid['entry_date'].dt.to_period('M').astype(str)
    month_stats = df_valid.groupby('month')['pnl'].agg(['count', 'sum', 'mean']).round(2)
    
    # Por hora
    df_valid['hour'] = df_valid['entry_date'].dt.hour
    hour_stats = df_valid.groupby('hour')['pnl'].agg(['count', 'sum', 'mean']).round(2)
    
    # Converter DataFrames para dicionários JSON-serializáveis
    def convert_stats_to_dict(stats_df):
        result = {}
        for index, row in stats_df.iterrows():
            # Garantir que o índice seja string
            key = str(index)
            result[key] = {
                'count': int(row['count']) if pd.notna(row['count']) else 0,
                'sum': float(row['sum']) if pd.notna(row['sum']) else 0.0,
                'mean': float(row['mean']) if pd.notna(row['mean']) else 0.0
            }
        return result
    
    return {
        "day_of_week": convert_stats_to_dict(day_stats),
        "monthly": convert_stats_to_dict(month_stats),
        "hourly": convert_stats_to_dict(hour_stats)
    }

# Função auxiliar para garantir que todos os valores sejam JSON-serializáveis
def make_json_serializable(obj):
    """Converte objetos pandas/numpy para tipos Python nativos"""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Period, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        # Tratar valores infinitos
        if np.isinf(obj):
            return None  # Retornar None em vez de Infinity
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.bytes_):
        return obj.decode('utf-8')
    elif pd.isna(obj) or obj is None:
        return None
    elif hasattr(obj, 'item'):  # Para outros tipos numpy que têm método item()
        item_value = obj.item()
        # Tratar valores infinitos também aqui
        if isinstance(item_value, float) and np.isinf(item_value):
            return None
        return item_value
    elif isinstance(obj, float):
        # Tratar valores infinitos para floats Python também
        if np.isinf(obj):
            return None
        return obj
    else:
        return obj

# Versão atualizada das outras funções de estatísticas para garantir serialização
def calcular_estatisticas_gerais(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estatísticas gerais das trades com serialização JSON correta"""
    if df.empty:
        return {}
    
    # Filtrar trades válidas
    df_valid = df.dropna(subset=['pnl'])
    
    total_trades = len(df_valid)
    if total_trades == 0:
        return {}
    
    # Resultados básicos
    total_pnl = df_valid['pnl'].sum()
    winning_trades = len(df_valid[df_valid['pnl'] > 0])
    losing_trades = len(df_valid[df_valid['pnl'] < 0])
    break_even_trades = len(df_valid[df_valid['pnl'] == 0])
    
    # Win rate
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Médias
    avg_win = df_valid[df_valid['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_valid[df_valid['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    avg_trade = df_valid['pnl'].mean()
    
    # Máximos e mínimos
    best_trade = df_valid['pnl'].max()
    worst_trade = df_valid['pnl'].min()
    
    # Profit Factor
    gross_profit = df_valid[df_valid['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_valid[df_valid['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else None
    
    # Expectativa
    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
    
    # Drawdown (se disponível)
    max_drawdown = df_valid['drawdown'].min() if 'drawdown' in df_valid.columns else 0
    
    # Criar resultado e garantir serialização JSON
    resultado = {
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "break_even_trades": int(break_even_trades),
        "win_rate": float(round(win_rate, 2)),
        "total_pnl": float(round(total_pnl, 2)),
        "avg_win": float(round(avg_win, 2)),
        "avg_loss": float(round(avg_loss, 2)),
        "avg_trade": float(round(avg_trade, 2)),
        "best_trade": float(round(best_trade, 2)),
        "worst_trade": float(round(worst_trade, 2)),
        "profit_factor": float(round(profit_factor, 2)) if profit_factor is not None else None,
        "expectancy": float(round(expectancy, 2)),
        "gross_profit": float(round(gross_profit, 2)),
        "gross_loss": float(round(gross_loss, 2)),
        "max_drawdown": float(round(max_drawdown, 2))
    }
    
    return make_json_serializable(resultado)

def calcular_estatisticas_por_ativo(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estatísticas agrupadas por ativo com serialização JSON correta"""
    if df.empty or 'symbol' not in df.columns:
        return {}
    
    stats_by_asset = {}
    
    for symbol in df['symbol'].unique():
        if pd.isna(symbol):
            continue
            
        asset_df = df[df['symbol'] == symbol].dropna(subset=['pnl'])
        
        if len(asset_df) == 0:
            continue
            
        stats_by_asset[str(symbol)] = {
            "total_trades": int(len(asset_df)),
            "total_pnl": float(round(asset_df['pnl'].sum(), 2)),
            "win_rate": float(round((len(asset_df[asset_df['pnl'] > 0]) / len(asset_df)) * 100, 2)),
            "avg_trade": float(round(asset_df['pnl'].mean(), 2)),
            "best_trade": float(round(asset_df['pnl'].max(), 2)),
            "worst_trade": float(round(asset_df['pnl'].min(), 2))
        }
    
    return make_json_serializable(stats_by_asset)

def calcular_custos_operacionais(df: pd.DataFrame, taxa_corretagem: float = 0.5, taxa_emolumentos: float = 0.03) -> Dict[str, Any]:
    """Calcula custos operacionais estimados"""
    if df.empty:
        return {}
    
    df_valid = df.dropna(subset=['entry_price', 'exit_price'])
    total_trades = len(df_valid)
    
    # Calcular valor total operado
    df_valid['valor_entrada'] = df_valid['entry_price'] * df_valid.get('quantity', 1)
    df_valid['valor_saida'] = df_valid['exit_price'] * df_valid.get('quantity', 1)
    valor_total_operado = (df_valid['valor_entrada'] + df_valid['valor_saida']).sum()
    
    # Custos estimados
    custo_corretagem = total_trades * taxa_corretagem  # Taxa fixa por operação
    custo_emolumentos = valor_total_operado * (taxa_emolumentos / 100)  # Taxa percentual
    custo_total = custo_corretagem + custo_emolumentos
    
    return {
        "total_trades": total_trades,
        "valor_total_operado": round(valor_total_operado, 2),
        "custo_corretagem": round(custo_corretagem, 2),
        "custo_emolumentos": round(custo_emolumentos, 2),
        "custo_total": round(custo_total, 2),
        "custo_por_trade": round(custo_total / total_trades, 2) if total_trades > 0 else 0
    }

# ============ FUNÇÕES PARA MÉTRICAS DIÁRIAS ============

def calcular_metricas_diarias(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas diárias baseadas nas trades"""
    if df.empty:
        return {}
    
    # Filtrar trades válidas
    df_valid = df.dropna(subset=['pnl', 'entry_date'])
    
    if df_valid.empty:
        return {}
    
    # Agrupar por dia
    df_valid['date'] = df_valid['entry_date'].dt.date
    daily_stats = df_valid.groupby('date').agg({
        'pnl': ['sum', 'count', 'mean'],
    }).round(2)
    
    daily_stats.columns = ['total_pnl', 'total_trades', 'avg_pnl']
    daily_stats['win_rate'] = df_valid.groupby('date').apply(
        lambda x: (x['pnl'] > 0).sum() / len(x) * 100
    ).round(2)
    
    # Calcular sequências de dias
    daily_stats['is_winner'] = daily_stats['total_pnl'] > 0
    daily_stats['is_loser'] = daily_stats['total_pnl'] < 0
    
    # Calcular drawdown
    daily_stats['cumulative_pnl'] = daily_stats['total_pnl'].cumsum()
    daily_stats['running_max'] = daily_stats['cumulative_pnl'].expanding().max()
    daily_stats['drawdown'] = daily_stats['cumulative_pnl'] - daily_stats['running_max']
    

    return daily_stats

def calcular_metricas_diarias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas diárias baseadas nas trades com drawdown correto
    CORRIGIDO: Sempre agrupa por data antes de calcular métricas diárias
    """
    if df.empty:
        print("⚠️ DataFrame vazio para cálculo de métricas diárias")
        return pd.DataFrame()
    
    print(f"🔍 DEBUG - calcular_metricas_diarias:")
    print(f"  Total de trades: {len(df)}")
    print(f"  Colunas disponíveis: {df.columns.tolist()}")
    
    # Filtrar trades válidas e ordenar por data
    df_valid = df.dropna(subset=['pnl', 'entry_date']).copy()
    df_valid = df_valid.sort_values('entry_date').reset_index(drop=True)
    
    print(f"  Trades válidas após filtro: {len(df_valid)}")
    
    if df_valid.empty:
        print("⚠️ Nenhuma trade válida encontrada")
        return pd.DataFrame()
    
    # Verificar se temos as colunas necessárias
    if 'pnl' not in df_valid.columns:
        print("❌ Coluna 'pnl' não encontrada. Colunas disponíveis:", df_valid.columns.tolist())
        return pd.DataFrame()
    
    if 'entry_date' not in df_valid.columns:
        print("❌ Coluna 'entry_date' não encontrada. Colunas disponíveis:", df_valid.columns.tolist())
        return pd.DataFrame()
    
    # CORREÇÃO 1: Garantir que sempre agrupamos por data
    df_valid['date'] = pd.to_datetime(df_valid['entry_date']).dt.date
    print(f"  Datas únicas encontradas: {df_valid['date'].nunique()}")
    print(f"  Primeira data: {df_valid['date'].min()}")
    print(f"  Última data: {df_valid['date'].max()}")
    
    # CORREÇÃO 2: Calcular saldo cumulativo por dia (não por trade)
    df_valid['saldo_cumulativo'] = df_valid['pnl'].cumsum()
    df_valid['saldo_maximo'] = df_valid['saldo_cumulativo'].cummax()
    df_valid['drawdown_trade'] = df_valid['saldo_cumulativo'] - df_valid['saldo_maximo']
    
    # CORREÇÃO 3: Agrupar por dia ANTES de calcular estatísticas
    daily_stats = df_valid.groupby('date').agg({
        'pnl': ['sum', 'count', 'mean'],
        'saldo_cumulativo': 'last',  # Saldo final do dia
        'saldo_maximo': 'last',      # Pico até o final do dia
        'drawdown_trade': 'min'      # Pior drawdown do dia
    }).round(2)
    
    # Simplificar nomes das colunas
    daily_stats.columns = ['total_pnl', 'total_trades', 'avg_pnl', 'saldo_final', 'peak_final', 'drawdown_dia']
    
    # CORREÇÃO 4: Calcular win rate diário baseado no PnL consolidado do dia
    daily_stats['is_winner'] = daily_stats['total_pnl'] > 0
    daily_stats['is_loser'] = daily_stats['total_pnl'] < 0
    
    # CORREÇÃO 5: Calcular drawdown correto para o dia (baseado no saldo final vs pico final)
    daily_stats['drawdown'] = daily_stats['saldo_final'] - daily_stats['peak_final']
    
    # CORREÇÃO 6: Calcular máximo histórico e drawdown cumulativo por dia
    daily_stats['running_max'] = daily_stats['saldo_final'].cummax()
    daily_stats['drawdown_cumulativo'] = daily_stats['saldo_final'] - daily_stats['running_max']
    
    # PADRONIZAÇÃO: Usar função centralizada para calcular drawdown
    drawdown_data = calcular_drawdown_padronizado(df)
    max_drawdown_trades = drawdown_data["max_drawdown"]
    max_drawdown_pct_trades = drawdown_data["max_drawdown_pct"]
    
    # Logs de debug para verificar padronização
    print(f"  PADRONIZAÇÃO - Drawdown máximo (trades): R$ {max_drawdown_trades:.2f} ({max_drawdown_pct_trades:.2f}%)")
    print(f"  PADRONIZAÇÃO - Drawdown máximo (dias): R$ {abs(daily_stats['drawdown_cumulativo'].min()):.2f}")
    print(f"  PADRONIZAÇÃO - Verificação: valores devem ser iguais")
    
    # Logs de debug detalhados
    print(f"  Dias com resultado positivo: {len(daily_stats[daily_stats['total_pnl'] > 0])}")
    print(f"  Dias com resultado negativo: {len(daily_stats[daily_stats['total_pnl'] < 0])}")
    print(f"  Maior ganho diário: {daily_stats['total_pnl'].max()}")
    print(f"  Maior perda diária: {daily_stats['total_pnl'].min()}")
    print(f"  Média de trades por dia: {daily_stats['total_trades'].mean():.1f}")
    print(f"  Total de dias operados: {len(daily_stats)}")
    
    # Verificar se os dados estão corretos
    print(f"  Verificação - Soma de PnL diário: {daily_stats['total_pnl'].sum()}")
    print(f"  Verificação - Soma de PnL original: {df_valid['pnl'].sum()}")
    
    return daily_stats.reset_index()


def calcular_metricas_principais(df: pd.DataFrame, taxa_juros_mensal: float = 0.01, capital_inicial: float = None) -> Dict[str, Any]:
    """
    Calcula as métricas principais do dashboard
    CORRIGIDO: Usa a mesma lógica de drawdown das outras funções
    E SHARPE RATIO com fórmula específica
    """
    if df.empty:
        return {}
    
    # Usar a função de métricas diárias corrigida
    daily_stats = calcular_metricas_diarias(df)
    
    if daily_stats.empty:
        return {}
    
    # Calcular métricas globais usando os mesmos campos das outras funções
    df_valid = df.dropna(subset=['pnl', 'entry_date']).copy()
    df_valid = df_valid.sort_values('entry_date').reset_index(drop=True)
    
    # Calcular saldo cumulativo (igual às outras funções)
    df_valid['Saldo'] = df_valid['pnl'].cumsum()
    df_valid['Saldo_Maximo'] = df_valid['Saldo'].cummax()
    df_valid['Drawdown'] = df_valid['Saldo'] - df_valid['Saldo_Maximo']
    
    # Métricas gerais
    total_pnl = df_valid['pnl'].sum()
    total_trades = len(df_valid)
    winning_trades = len(df_valid[df_valid['pnl'] > 0])
    losing_trades = len(df_valid[df_valid['pnl'] < 0])
    
    # Payoff Ratio (Ganho médio / Perda média)
    avg_win = df_valid[df_valid['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(df_valid[df_valid['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    # PADRONIZADO: Usar função centralizada para calcular drawdown
    drawdown_data = calcular_drawdown_padronizado(df)
    max_drawdown = drawdown_data["max_drawdown"]
    max_drawdown_pct = drawdown_data["max_drawdown_pct"]
    saldo_final = drawdown_data["saldo_final"]
    capital_inicial = drawdown_data["capital_inicial"]
    
    # CALCULAR DD MÉDIO - CORREÇÃO ADICIONADA
    # Calcular drawdown médio baseado nos trades individuais
    equity = df_valid['pnl'].cumsum()
    peak = equity.cummax()
    drawdown_series = equity - peak
    drawdown_values = drawdown_series[drawdown_series < 0].abs()  # Apenas valores negativos (drawdowns)
    avg_drawdown = drawdown_values.mean() if len(drawdown_values) > 0 else 0
    
    # CAPITAL INICIAL CORRIGIDO
    # Se não fornecido, calcular baseado no drawdown máximo
    if capital_inicial is None:
        # Método 1: Baseado no fato de que drawdown% = drawdown$ / saldo_final
        # Se drawdown% = 66.22% e drawdown$ = 835.8
        # Então: saldo_final = drawdown$ / (drawdown% / 100)
        saldo_final = df_valid['Saldo'].iloc[-1]  # 1262.2
        
        # Para calcular capital inicial, usar: capital = saldo_final + abs(saldo_minimo)
        saldo_minimo = df_valid['Saldo'].min()  # Ponto mais baixo
        capital_estimado = saldo_final + abs(saldo_minimo) if saldo_minimo < 0 else saldo_final + max_drawdown
        
        # Método alternativo: usar drawdown 3x como base mínima
        capital_por_drawdown = max_drawdown * 3  # 835.8 * 3 = 2507.4
        
        # Usar o maior entre os dois métodos para ser conservador
        capital_inicial = max(capital_estimado, capital_por_drawdown)
    
    # SHARPE RATIO CORRIGIDO - Usar mesma fórmula do FunCalculos.py
    # Calcular retornos dos trades individuais (como no FunCalculos.py)
    returns = df_valid['pnl'].values
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
    cdi = 0.12  # Taxa anual (12% ao ano) - mesma do FunCalculos.py
    sharpe_ratio = ((mean_return - cdi) / std_return) if std_return != 0 else 0
    
    # Fator de Recuperação
    recovery_factor = total_pnl / max_drawdown if max_drawdown != 0 else 0
    
    # Dias operados
    days_traded = len(daily_stats)
    
    # Estatísticas diárias CORRIGIDAS - baseadas em dias, não em operações
    winning_days = len(daily_stats[daily_stats['total_pnl'] > 0])
    losing_days = len(daily_stats[daily_stats['total_pnl'] < 0])
    daily_win_rate = (winning_days / days_traded * 100) if days_traded > 0 else 0
    
    # Ganhos e perdas diárias CORRIGIDOS - baseados em dias, não em operações
    daily_avg_win = daily_stats[daily_stats['total_pnl'] > 0]['total_pnl'].mean() if winning_days > 0 else 0
    daily_avg_loss = abs(daily_stats[daily_stats['total_pnl'] < 0]['total_pnl'].mean()) if losing_days > 0 else 0
    daily_max_win = daily_stats['total_pnl'].max() if not daily_stats.empty else 0
    daily_max_loss = daily_stats['total_pnl'].min() if not daily_stats.empty else 0  # Já é negativo
    
    # Média de operações por dia
    avg_trades_per_day = total_trades / days_traded if days_traded > 0 else 0
    
    # Sequências consecutivas
    consecutive_wins, consecutive_losses = calcular_sequencias_consecutivas(daily_stats)
    
    # Debug logs para verificar os cálculos
    print(f"🔍 DEBUG - Métricas diárias:")
    print(f"  Dias operados: {days_traded}")
    print(f"  Dias vencedores: {winning_days}")
    print(f"  Dias perdedores: {losing_days}")
    print(f"  Taxa de acerto diária: {daily_win_rate}%")
    print(f"  Ganho médio diário: {daily_avg_win}")
    print(f"  Perda média diária: {daily_avg_loss}")
    print(f"  Ganho máximo diário: {daily_max_win}")
    print(f"  Perda máxima diária: {daily_max_loss}")
    print(f"  Operações por dia: {avg_trades_per_day}")
    print(f"  DD Médio: {avg_drawdown:.2f}")
    print(f"  Sharpe Ratio (corrigido): {sharpe_ratio:.2f}")
    
    return {
        "metricas_principais": {
            "sharpe_ratio": round(sharpe_ratio, 2),  # PADRONIZADO - mesma fórmula do FunCalculos.py
            "fator_recuperacao": round(recovery_factor, 2),
            "drawdown_maximo": round(-max_drawdown, 2),  # Negativo para compatibilidade
            "drawdown_maximo_pct": round(max_drawdown_pct, 2),
            "drawdown_medio": round(avg_drawdown, 2),  # NOVO: DD Médio calculado
            "dias_operados": int(days_traded),
            "resultado_liquido": round(total_pnl, 2),
            # PADRONIZAÇÃO: Usar drawdown calculado com trades individuais (mesmo valor do original)
            "drawdown_maximo_padronizado": round(-max_drawdown, 2),  # Negativo para compatibilidade
            "drawdown_maximo_pct_padronizado": round(max_drawdown_pct, 2),
            # PADRONIZAÇÃO: Valores para API (positivos)
            "max_drawdown_padronizado": round(max_drawdown, 2),  # Valor positivo para API
            "max_drawdown_pct_padronizado": round(max_drawdown_pct, 2),  # Percentual para API
            # Campos adicionais para debug/transparência
            "capital_estimado": round(capital_inicial, 2)
        },
        "ganhos_perdas": {
            "ganho_medio_diario": round(daily_avg_win, 2),
            "perda_media_diaria": round(daily_avg_loss, 2),
            "payoff_diario": round(daily_avg_win / daily_avg_loss if daily_avg_loss != 0 else 0, 2),
            "ganho_maximo_diario": round(daily_max_win, 2),
            "perda_maxima_diaria": round(abs(daily_max_loss), 2)  # Valor absoluto para compatibilidade
        },
        "estatisticas_operacao": {
            "media_operacoes_dia": round(avg_trades_per_day, 1),
            "taxa_acerto_diaria": round(daily_win_rate, 2),
            "dias_vencedores_perdedores": f"{winning_days} / {losing_days}",
            "dias_perdedores_consecutivos": consecutive_losses,
            "dias_vencedores_consecutivos": consecutive_wins
        }
    }

def calcular_sharpe_ratio_customizado(total_pnl: float, max_drawdown: float, periodo_meses: float, taxa_juros_mensal: float = 0.01, capital_inicial: float = None) -> Dict[str, float]:
    """
    Calcula o Sharpe Ratio usando a fórmula específica fornecida
    
    Args:
        total_pnl: Lucro/prejuízo total
        max_drawdown: Drawdown máximo (valor positivo)
        periodo_meses: Período em meses
        taxa_juros_mensal: Taxa de juros mensal (padrão 1% = 0.01)
        capital_inicial: Capital inicial (se None, será estimado)
    
    Returns:
        Dict com os componentes do cálculo e o resultado final
    """
    
    # Estimar capital inicial se não fornecido
    if capital_inicial is None:
        capital_inicial = max(max_drawdown * 3, abs(total_pnl) * 2, 100000)
    
    # Taxa de juros do período
    taxa_juros_periodo = taxa_juros_mensal * periodo_meses
    
    # Rentabilidade do período em percentual
    rentabilidade_periodo_pct = (total_pnl / capital_inicial) * 100
    
    # Numerador: (Rentabilidade período - taxa de juros período)
    numerador = rentabilidade_periodo_pct - (taxa_juros_periodo * 100)
    
    # Denominador: Risco (drawdown / 3x drawdown)
    drawdown_3x = max_drawdown * 3
    risco_pct = (max_drawdown / drawdown_3x) * 100 if drawdown_3x > 0 else 33.33  # Valor padrão em vez de 100
    
    # Sharpe Ratio
    sharpe_ratio = numerador / risco_pct if risco_pct != 0 and risco_pct != 33.33 else 0
    
    return {
        "sharpe_ratio": round(sharpe_ratio, 2),
        "total_pnl": total_pnl,
        "capital_inicial": capital_inicial,
        "rentabilidade_pct": round(rentabilidade_periodo_pct, 2),
        "taxa_juros_periodo_pct": round(taxa_juros_periodo * 100, 2),
        "numerador": round(numerador, 2),
        "max_drawdown": max_drawdown,
        "drawdown_3x": drawdown_3x,
        "risco_pct": round(risco_pct, 2),
        "periodo_meses": periodo_meses
    }


def calcular_sequencias_consecutivas(daily_stats: pd.DataFrame) -> Tuple[int, int]:
    """Calcula sequências consecutivas de dias vencedores e perdedores"""
    if daily_stats.empty:
        return 0, 0
    
    # Sequências de vitórias
    wins = daily_stats['is_winner'].astype(int)
    win_sequences = []
    current_sequence = 0
    
    for win in wins:
        if win:
            current_sequence += 1
        else:
            if current_sequence > 0:
                win_sequences.append(current_sequence)
            current_sequence = 0
    if current_sequence > 0:
        win_sequences.append(current_sequence)
    
    # Sequências de perdas
    losses = daily_stats['is_loser'].astype(int)
    loss_sequences = []
    current_sequence = 0
    
    for loss in losses:
        if loss:
            current_sequence += 1
        else:
            if current_sequence > 0:
                loss_sequences.append(current_sequence)
            current_sequence = 0
    if current_sequence > 0:
        loss_sequences.append(current_sequence)
    
    max_consecutive_wins = max(win_sequences) if win_sequences else 0
    max_consecutive_losses = max(loss_sequences) if loss_sequences else 0
    
    return max_consecutive_wins, max_consecutive_losses
# Adicione ao seu main.py

import pandas as pd
import numpy as np
from typing import Dict, Any
from flask import Flask, request, jsonify

def calcular_disciplina_completa(df: pd.DataFrame, fator_disciplina: float = 0.2, multiplicador_furia: float = 2.0) -> Dict[str, Any]:
    """
    Calcula TODOS os índices de disciplina em uma função única:
    - Disciplina Stop (por operação)
    - Disciplina Perda/Dia (por dia)
    - Métrica de Fúria Diária (baseada em múltiplo da perda média)
    
    Args:
        df: DataFrame com as operações
        fator_disciplina: Fator para calcular meta máxima (padrão 20% = 0.2)
        multiplicador_furia: Multiplicador para definir "dia de fúria" (padrão 2.0 = 2x a perda média)
    
    Returns:
        Dict com todas as métricas de disciplina (JSON serializable)
    """
    if df.empty:
        return {"error": "DataFrame vazio"}
    
    # Encontrar colunas corretas
    resultado_col = None
    data_col = None
    quantidade_col = None
    
    for col_name in ['operation_result', 'pnl', 'resultado']:
        if col_name in df.columns:
            resultado_col = col_name
            break
    
    for col_name in ['entry_date', 'data_abertura', 'data']:
        if col_name in df.columns:
            data_col = col_name
            break
    
    for col_name in ['qty_buy', 'Quantidade', 'qtd', 'qty', 'volume', 'contratos', 'acoes', 'size']:
        if col_name in df.columns:
            quantidade_col = col_name
            break
    
    if resultado_col is None or data_col is None:
        return {"error": "Colunas de resultado ou data não encontradas"}
    
    # Quantidade é opcional
    quantidade_disponivel = quantidade_col is not None
    
    # Filtrar operações válidas
    if quantidade_disponivel:
        df_valid = df.dropna(subset=[resultado_col, data_col, quantidade_col]).copy()
    else:
        df_valid = df.dropna(subset=[resultado_col, data_col]).copy()
    
    if df_valid.empty:
        return {"error": "Nenhuma operação válida encontrada"}
    
    # Converter data para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df_valid[data_col]):
        df_valid[data_col] = pd.to_datetime(df_valid[data_col])
    
    # ===== VARIÁVEIS GERAIS =====
    total_operacoes = int(len(df_valid))
    
    # ===== DISCIPLINA ALAVANCAGEM =====
    if quantidade_disponivel:
        # Calcular média de quantidade
        media_quantidade = float(df_valid[quantidade_col].mean())
        limite_alavancagem = media_quantidade * 2  # 2x a média de quantidade
        
        # Identificar operações que ultrapassaram 2x a média
        operacoes_alavancadas = df_valid[df_valid[quantidade_col] > limite_alavancagem]
        qtd_operacoes_alavancadas = int(len(operacoes_alavancadas))
        total_operacoes_quantidade = int(len(df_valid))
        
        # Calcular índice de disciplina de alavancagem
        indice_disciplina_alavancagem = (1 - (qtd_operacoes_alavancadas / total_operacoes_quantidade)) * 100
        
        disciplina_alavancagem = {
            "disponivel": True,
            "total_operacoes": total_operacoes_quantidade,
            "media_quantidade": round(media_quantidade, 2),
            "limite_alavancagem": round(limite_alavancagem, 2),
            "operacoes_alavancadas": qtd_operacoes_alavancadas,
            "operacoes_dentro_limite": total_operacoes_quantidade - qtd_operacoes_alavancadas,
            "indice_disciplina_alavancagem": round(indice_disciplina_alavancagem, 2),
            "detalhes_alavancagem": [
                {
                    "operacao": i + 1,
                    "quantidade": int(row[quantidade_col]),
                    "excesso_limite": round(float(row[quantidade_col]) - limite_alavancagem, 2),
                    "multiplo_media": round(float(row[quantidade_col]) / media_quantidade, 2),
                    "data": row[data_col].strftime('%d/%m/%Y'),
                    "resultado": round(float(row[resultado_col]), 2)
                }
                for i, (_, row) in enumerate(operacoes_alavancadas.iterrows())
            ] if qtd_operacoes_alavancadas > 0 else []
        }
    else:
        disciplina_alavancagem = {
            "disponivel": False,
            "motivo": "Coluna de quantidade não encontrada",
            "colunas_procuradas": ['Qtd Compra', 'Quantidade', 'qtd', 'qty', 'volume', 'contratos', 'acoes', 'size']
        }
    
    # ===== PREPARAR DADOS DIÁRIOS =====
    df_valid['Data'] = df_valid[data_col].dt.date
    
    # Agrupar por dia
    resultado_diario = df_valid.groupby('Data').agg({
        resultado_col: ['sum', 'count', 'min']
    }).round(2)
    
    resultado_diario.columns = ['PnL_Dia', 'Trades_Dia', 'Pior_Trade_Dia']
    resultado_diario = resultado_diario.reset_index()
    
    # Separar dias com perda
    dias_com_perda = resultado_diario[resultado_diario['PnL_Dia'] < 0].copy()
    
    # ===== NOVA MÉTRICA: FÚRIA DIÁRIA =====
    if dias_com_perda.empty:
        furia_diaria = {
            "disponivel": False,
            "motivo": "Não há dias com perda para calcular fúria",
            "dias_com_perda": 0,
            "perda_media_diaria": 0.0,
            "limite_furia": 0.0,
            "dias_furia": 0,
            "total_dias_operados": int(len(resultado_diario)),
            "percentual_dias_furia": 0.0,
            "frequencia_furia": 0.0,
            "detalhes_furia": []
        }
    else:
        # Calcular perda média diária
        perda_media_diaria = float(abs(dias_com_perda['PnL_Dia'].mean()))
        
        # Definir limite de fúria (multiplicador da perda média)
        limite_furia = perda_media_diaria * multiplicador_furia
        
        # Identificar dias de fúria (perdas maiores que o limite)
        dias_furia = dias_com_perda[abs(dias_com_perda['PnL_Dia']) > limite_furia]
        qtd_dias_furia = int(len(dias_furia))
        
        # Calcular métricas
        total_dias_operados = int(len(resultado_diario))  # Total de dias que teve operações
        percentual_dias_furia = (qtd_dias_furia / total_dias_operados) * 100  # % em relação aos dias operados
        frequencia_furia = (qtd_dias_furia / len(dias_com_perda)) * 100  # Em relação aos dias com perda
        
        furia_diaria = {
            "disponivel": True,
            "dias_com_perda": int(len(dias_com_perda)),
            "perda_media_diaria": round(perda_media_diaria, 2),
            "limite_furia": round(limite_furia, 2),
            "multiplicador_usado": multiplicador_furia,
            "dias_furia": qtd_dias_furia,
            "total_dias_operados": total_dias_operados,
            "percentual_dias_furia": round(percentual_dias_furia, 2),
            "frequencia_furia_vs_dias_perda": round(frequencia_furia, 2),
            "detalhes_furia": [
                {
                    "data": row['Data'].strftime('%d/%m/%Y'),
                    "pnl_dia": round(float(row['PnL_Dia']), 2),
                    "perda_absoluta": round(abs(float(row['PnL_Dia'])), 2),
                    "trades_dia": int(row['Trades_Dia']),
                    "excesso_limite": round(abs(float(row['PnL_Dia'])) - limite_furia, 2),
                    "multiplo_media": round(abs(float(row['PnL_Dia'])) / perda_media_diaria, 2),
                    "pior_trade": round(float(row['Pior_Trade_Dia']), 2),
                    "intensidade": "extrema" if abs(float(row['PnL_Dia'])) > limite_furia * 1.5 else "alta"
                }
                for _, row in dias_furia.iterrows()
            ] if qtd_dias_furia > 0 else [],
            "estatisticas_intensidade": {
                "furia_alta": int(len(dias_furia[abs(dias_furia['PnL_Dia']) <= limite_furia * 1.5])),
                "furia_extrema": int(len(dias_furia[abs(dias_furia['PnL_Dia']) > limite_furia * 1.5])),
                "pior_dia_furia": round(float(dias_furia['PnL_Dia'].min()), 2) if qtd_dias_furia > 0 else 0.0,
                "media_perda_furia": round(float(dias_furia['PnL_Dia'].mean()), 2) if qtd_dias_furia > 0 else 0.0
            }
        }
    
    # ===== PROBABILIDADE DE FÚRIA (SEQUENCIAL) =====
    # Calcular sequências de perdas consecutivas
    df_valid['eh_perda'] = df_valid[resultado_col] < 0
    df_valid = df_valid.sort_values(data_col).reset_index(drop=True)
    
    # Identificar sequências de perdas
    sequencias_perdas = []
    sequencia_atual = 0
    
    for eh_perda in df_valid['eh_perda']:
        if eh_perda:
            sequencia_atual += 1
        else:
            if sequencia_atual > 0:
                sequencias_perdas.append(sequencia_atual)
                sequencia_atual = 0
    
    # Adicionar última sequência se terminou em perda
    if sequencia_atual > 0:
        sequencias_perdas.append(sequencia_atual)
    
    if sequencias_perdas:
        maior_sequencia_perdas = max(sequencias_perdas)
        total_sequencias = len(sequencias_perdas)
        media_sequencia_perdas = sum(sequencias_perdas) / len(sequencias_perdas)
        
        # Calcular probabilidade de "fúria" (sequência >= 3 perdas)
        sequencias_furia = [s for s in sequencias_perdas if s >= 3]
        qtd_episodios_furia = len(sequencias_furia)
        
        # Probabilidade = episódios de fúria / total de sequências de perda
        if total_sequencias > 0:
            probabilidade_furia = (qtd_episodios_furia / total_sequencias) * 100
        else:
            probabilidade_furia = 0.0
        
        # Calcular frequência de fúria por total de trades
        frequencia_furia_trades = (qtd_episodios_furia / total_operacoes) * 100
        
        probabilidade_furia_resultado = {
            "disponivel": True,
            "total_operacoes": total_operacoes,
            "total_operacoes_perdedoras": int(len(df_valid[df_valid['eh_perda']])),
            "total_sequencias_perda": total_sequencias,
            "maior_sequencia_perdas": maior_sequencia_perdas,
            "media_sequencia_perdas": round(media_sequencia_perdas, 2),
            "episodios_furia": qtd_episodios_furia,
            "probabilidade_furia": round(probabilidade_furia, 2),
            "frequencia_furia_por_trades": round(frequencia_furia_trades, 2),
            "detalhes_sequencias": [
                {
                    "sequencia_numero": i + 1,
                    "tamanho_sequencia": seq,
                    "eh_furia": seq >= 3,
                    "classificacao": "fúria" if seq >= 3 else "normal" if seq <= 2 else "moderada"
                }
                for i, seq in enumerate(sequencias_perdas)
            ],
            "estatisticas_sequencias": {
                "sequencias_1_perda": len([s for s in sequencias_perdas if s == 1]),
                "sequencias_2_perdas": len([s for s in sequencias_perdas if s == 2]),
                "sequencias_3_ou_mais": len([s for s in sequencias_perdas if s >= 3]),
                "sequencias_5_ou_mais": len([s for s in sequencias_perdas if s >= 5])
            }
        }
    else:
        probabilidade_furia_resultado = {
            "disponivel": True,
            "total_operacoes": total_operacoes,
            "total_operacoes_perdedoras": 0,
            "total_sequencias_perda": 0,
            "maior_sequencia_perdas": 0,
            "media_sequencia_perdas": 0.0,
            "episodios_furia": 0,
            "probabilidade_furia": 0.0,
            "frequencia_furia_por_trades": 0.0,
            "detalhes_sequencias": [],
            "estatisticas_sequencias": {
                "sequencias_1_perda": 0,
                "sequencias_2_perdas": 0,
                "sequencias_3_ou_mais": 0,
                "sequencias_5_ou_mais": 0
            }
        }
    
    # ===== DISCIPLINA STOP (POR OPERAÇÃO) =====
    operacoes_perdedoras = df_valid[df_valid[resultado_col] < 0].copy()
    
    if operacoes_perdedoras.empty:
        disciplina_operacao = {
            "operacoes_perdedoras": 0,
            "media_perda": 0.0,
            "meta_maxima_perda": 0.0,
            "operacoes_excederam_meta": 0,
            "indice_disciplina": 100.0,
            "operacoes_dentro_meta": 0,
            "detalhes_excesso": []
        }
    else:
        # Calcular disciplina por operação
        media_perda = float(operacoes_perdedoras[resultado_col].mean())
        meta_maxima_perda = media_perda + (media_perda * fator_disciplina)
        
        operacoes_excederam = operacoes_perdedoras[operacoes_perdedoras[resultado_col] < meta_maxima_perda]
        num_operacoes_excederam = int(len(operacoes_excederam))
        operacoes_dentro_meta = int(len(operacoes_perdedoras) - num_operacoes_excederam)
        
        indice_disciplina_op = (operacoes_dentro_meta / len(operacoes_perdedoras)) * 100
        
        disciplina_operacao = {
            "operacoes_perdedoras": int(len(operacoes_perdedoras)),
            "media_perda": round(media_perda, 2),
            "meta_maxima_perda": round(meta_maxima_perda, 2),
            "operacoes_excederam_meta": num_operacoes_excederam,
            "indice_disciplina": round(indice_disciplina_op, 2),
            "operacoes_dentro_meta": operacoes_dentro_meta,
            "detalhes_excesso": [
                {
                    "operacao": i + 1,
                    "resultado": round(float(row[resultado_col]), 2),
                    "excesso": round(float(row[resultado_col]) - meta_maxima_perda, 2)
                }
                for i, (_, row) in enumerate(operacoes_excederam.iterrows())
            ] if num_operacoes_excederam > 0 else []
        }
    
    # ===== DISCIPLINA PERDA/DIA (MÉTODO ORIGINAL) =====
    if dias_com_perda.empty:
        disciplina_dia = {
            "dias_com_perda": 0,
            "media_perda_diaria": 0.0,
            "meta_maxima_perda_dia": 0.0,
            "dias_excederam_meta": 0,
            "indice_disciplina_diaria": 100.0,
            "dias_dentro_meta": 0,
            "detalhes_dias_excesso": []
        }
    else:
        # Calcular disciplina por dia
        media_perda_diaria = float(dias_com_perda['PnL_Dia'].mean())
        meta_maxima_perda_dia = media_perda_diaria + (media_perda_diaria * fator_disciplina)
        
        dias_excederam = dias_com_perda[dias_com_perda['PnL_Dia'] < meta_maxima_perda_dia]
        num_dias_excederam = int(len(dias_excederam))
        dias_dentro_meta = int(len(dias_com_perda) - num_dias_excederam)
        
        indice_disciplina_dia = (dias_dentro_meta / len(dias_com_perda)) * 100
        
        disciplina_dia = {
            "dias_com_perda": int(len(dias_com_perda)),
            "media_perda_diaria": round(media_perda_diaria, 2),
            "meta_maxima_perda_dia": round(meta_maxima_perda_dia, 2),
            "dias_excederam_meta": num_dias_excederam,
            "indice_disciplina_diaria": round(indice_disciplina_dia, 2),
            "dias_dentro_meta": dias_dentro_meta,
            "detalhes_dias_excesso": [
                {
                    "data": row['Data'].strftime('%d/%m/%Y'),
                    "pnl_dia": round(float(row['PnL_Dia']), 2),
                    "trades_dia": int(row['Trades_Dia']),
                    "excesso": round(float(row['PnL_Dia']) - meta_maxima_perda_dia, 2),
                    "pior_trade": round(float(row['Pior_Trade_Dia']), 2)
                }
                for _, row in dias_excederam.iterrows()
            ] if num_dias_excederam > 0 else []
        }
    
    # ===== ESTATÍSTICAS GERAIS =====
    total_dias = int(len(resultado_diario))
    dias_com_ganho = int(len(resultado_diario[resultado_diario['PnL_Dia'] > 0]))
    dias_breakeven = int(len(resultado_diario[resultado_diario['PnL_Dia'] == 0]))
    
    pior_operacao = float(df_valid[resultado_col].min())
    melhor_operacao = float(df_valid[resultado_col].max())
    pior_dia = float(resultado_diario['PnL_Dia'].min())
    melhor_dia = float(resultado_diario['PnL_Dia'].max())
    
    # ===== RESUMO COMPARATIVO =====
    resumo = {
        "disciplina_operacao": disciplina_operacao["indice_disciplina"],
        "disciplina_dia": disciplina_dia["indice_disciplina_diaria"],
        "disciplina_alavancagem": disciplina_alavancagem["indice_disciplina_alavancagem"] if disciplina_alavancagem["disponivel"] else None,
        "probabilidade_furia_sequencial": probabilidade_furia_resultado["probabilidade_furia"],
        "percentual_dias_furia": furia_diaria["percentual_dias_furia"] if furia_diaria["disponivel"] else 0.0,
        "frequencia_furia_diaria": furia_diaria["frequencia_furia_vs_dias_perda"] if furia_diaria["disponivel"] else 0.0,
        "diferenca_operacao_dia": round(disciplina_operacao["indice_disciplina"] - disciplina_dia["indice_disciplina_diaria"], 2),
        "melhor_disciplina": "operacao" if disciplina_operacao["indice_disciplina"] > disciplina_dia["indice_disciplina_diaria"] else "dia",
        "media_perda_operacao": disciplina_operacao["media_perda"],
        "media_perda_dia": disciplina_dia["media_perda_diaria"],
        "limite_furia_diaria": furia_diaria["limite_furia"] if furia_diaria["disponivel"] else None
    }
    
    # Adicionar comparação com alavancagem se disponível
    if disciplina_alavancagem["disponivel"]:
        resumo["diferenca_operacao_alavancagem"] = round(disciplina_operacao["indice_disciplina"] - disciplina_alavancagem["indice_disciplina_alavancagem"], 2)
        resumo["diferenca_dia_alavancagem"] = round(disciplina_dia["indice_disciplina_diaria"] - disciplina_alavancagem["indice_disciplina_alavancagem"], 2)
        
        # Encontrar a melhor disciplina entre todas
        disciplinas = {
            "operacao": disciplina_operacao["indice_disciplina"],
            "dia": disciplina_dia["indice_disciplina_diaria"],
            "alavancagem": disciplina_alavancagem["indice_disciplina_alavancagem"]
        }
        resumo["melhor_disciplina_geral"] = max(disciplinas, key=disciplinas.get)
        resumo["pior_disciplina_geral"] = min(disciplinas, key=disciplinas.get)
    
    # Adicionar indicadores de risco baseados na fúria
    resumo["risco_emocional_sequencial"] = "alto" if probabilidade_furia_resultado["probabilidade_furia"] > 50 else "medio" if probabilidade_furia_resultado["probabilidade_furia"] > 25 else "baixo"
    resumo["risco_emocional_diario"] = "alto" if furia_diaria["percentual_dias_furia"] > 15 else "medio" if furia_diaria["percentual_dias_furia"] > 5 else "baixo"
    resumo["maior_sequencia_perdas"] = probabilidade_furia_resultado["maior_sequencia_perdas"]
    
    # ===== RESULTADO FINAL =====
    return {
        "disciplina_operacao": disciplina_operacao,
        "disciplina_dia": disciplina_dia,
        "disciplina_alavancagem": disciplina_alavancagem,
        "probabilidade_furia_sequencial": probabilidade_furia_resultado,
        "furia_diaria": furia_diaria,
        "estatisticas_gerais": {
            "total_operacoes": total_operacoes,
            "total_dias": total_dias,
            "dias_com_ganho": dias_com_ganho,
            "dias_com_perda": disciplina_dia["dias_com_perda"],
            "dias_breakeven": dias_breakeven,
            "operacoes_ganhadoras": total_operacoes - disciplina_operacao["operacoes_perdedoras"],
            "operacoes_perdedoras": disciplina_operacao["operacoes_perdedoras"],
            "pior_operacao": round(pior_operacao, 2),
            "melhor_operacao": round(melhor_operacao, 2),
            "pior_dia": round(pior_dia, 2),
            "melhor_dia": round(melhor_dia, 2),
            "media_trades_por_dia": round(total_operacoes / total_dias, 1),
            "fator_disciplina_usado": float(fator_disciplina),
            "multiplicador_furia_usado": float(multiplicador_furia),
            "coluna_quantidade_encontrada": quantidade_col if quantidade_disponivel else None
        },
        "resumo_comparativo": resumo,
        "resultado_diario_completo": [
            {
                "data": row['Data'].strftime('%d/%m/%Y'),
                "pnl_dia": round(float(row['PnL_Dia']), 2),
                "trades_dia": int(row['Trades_Dia']),
                "pior_trade": round(float(row['Pior_Trade_Dia']), 2),
                "status": "ganho" if row['PnL_Dia'] > 0 else "perda" if row['PnL_Dia'] < 0 else "breakeven",
                "dentro_meta": bool(row['PnL_Dia'] >= disciplina_dia["meta_maxima_perda_dia"] if row['PnL_Dia'] < 0 else True),
                "eh_furia": bool(abs(row['PnL_Dia']) > furia_diaria["limite_furia"] if furia_diaria["disponivel"] and row['PnL_Dia'] < 0 else False)
            }
            for _, row in resultado_diario.iterrows()
        ]
    }

# ============ API ÚNICA SIMPLIFICADA PARA MÚLTIPLOS ARQUIVOS ============

@app.route('/api/disciplina-completa', methods=['POST'])
def api_disciplina_completa():
    """
    Endpoint ÚNICO para calcular TODAS as métricas de disciplina
    Suporta tanto um arquivo ('file') quanto múltiplos arquivos ('files')
    """
    try:
        # Parâmetros opcionais
        fator_disciplina = float(request.form.get('fator_disciplina', 0.2))
        multiplicador_furia = float(request.form.get('multiplicador_furia', 2.0))
        
        # Lista para armazenar todos os DataFrames
        dataframes = []
        arquivos_processados = []
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_safe(arquivo)
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_safe(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo enviado. Use 'file' para um arquivo ou 'files' para múltiplos"}), 400
        
        # Concatenar todos os DataFrames em um só
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # Calcular disciplina no DataFrame consolidado
        resultado = calcular_disciplina_completa(df_consolidado, fator_disciplina, multiplicador_furia)
        
        if 'error' in resultado:
            return jsonify(resultado), 400
        
        # Adicionar informações sobre os arquivos processados
        resultado['info_arquivos'] = {
            "total_arquivos": len(arquivos_processados),
            "nomes_arquivos": arquivos_processados,
            "total_registros_consolidados": len(df_consolidado)
        }
        
        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ============ FUNÇÃO AUXILIAR PARA DEBUG ============

def debug_json_serializable(obj, path=""):
    """
    Função para identificar valores não serializáveis em JSON
    """
    import json
    import numpy as np
    
    try:
        if isinstance(obj, dict):
            for key, value in obj.items():
                debug_json_serializable(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                debug_json_serializable(value, f"{path}[{i}]")
        else:
            # Tentar serializar o valor individual
            json.dumps(obj)
    except TypeError as e:
        print(f"Erro em {path}: {type(obj)} - {obj}")
        print(f"Erro: {e}")
        
        # Sugerir correção
        if isinstance(obj, np.bool_):
            print(f"Correção: bool({obj})")
        elif isinstance(obj, np.int64):
            print(f"Correção: int({obj})")
        elif isinstance(obj, np.float64):
            print(f"Correção: float({obj})")
        elif hasattr(obj, 'item'):
            print(f"Correção: {obj}.item()")

# ============ FUNÇÃO AUXILIAR PARA DEBUG ============


#Rota para receber o CSV e retornar as métricas
@app.route('/api/tabela-multipla', methods=['POST'])
def api_tabela_multipla():
    """
    Endpoint para processar múltiplos arquivos de backtest
    Garantindo que retorne TODOS os dados incluindo Equity Curve Data
    """
    try:
        # Lista para armazenar todos os DataFrames
        dataframes = []
        arquivos_processados = []
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_safe(arquivo)
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_safe(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo enviado. Use 'file' para um arquivo ou 'files' para múltiplos"}), 400
        
        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # Processar cada arquivo individualmente
        resultados_individuais = {}
        print(f"🔍 Processando {len(dataframes)} arquivos individualmente:")
        for i, (df, nome_arquivo) in enumerate(zip(dataframes, arquivos_processados)):
            try:
                print(f"  📁 Arquivo {i+1}/{len(dataframes)}: {nome_arquivo}")
                print(f"     📊 Registros: {len(df)}")
                print(f"     📅 Colunas: {list(df.columns)}")
                
                # Garantir que 'pnl' exista antes de qualquer cálculo
                try:
                    if 'pnl' not in df.columns:
                        if 'operation_result' in df.columns:
                            df['pnl'] = df['operation_result']
                        elif 'Res. Intervalo Bruto' in df.columns:
                            df['pnl'] = pd.to_numeric(df['Res. Intervalo Bruto'], errors='coerce')
                        elif 'Res. Intervalo' in df.columns:
                            df['pnl'] = pd.to_numeric(df['Res. Intervalo'], errors='coerce')
                    # Converter para numérico por segurança
                    if 'pnl' in df.columns:
                        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                except Exception as e:
                    print(f"⚠️ DEBUG: Falha ao garantir 'pnl' antes do debug_drawdown: {e}")

                # DEBUG: Verificar padronização do drawdown
                debug_drawdown_calculation(df)
                
                # Garantir que 'pnl' exista antes de calcular métricas
                if 'pnl' not in df.columns and 'operation_result' in df.columns:
                    df['pnl'] = df['operation_result']
                if 'pnl' in df.columns:
                    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')

                resultado_individual = processar_backtest_completo(df, capital_inicial=capital_inicial, cdi=cdi)

                # Garantir compatibilidade de chaves no resultado individual (para o frontend)
                try:
                    # Copiar em camelCase as seções principais
                    if 'Day of Week Analysis' in resultado_individual:
                        resultado_individual['day_of_week'] = resultado_individual['Day of Week Analysis']
                    if 'Monthly Analysis' in resultado_individual:
                        resultado_individual['monthly'] = resultado_individual['Monthly Analysis']
                    if 'Equity Curve Data' in resultado_individual:
                        resultado_individual['equity_curve_data'] = resultado_individual['Equity Curve Data']
                except Exception as e:
                    print(f"⚠️ DEBUG: Falha ao padronizar chaves camelCase: {e}")
                
                if 'equity_curve_data' not in resultado_individual:
                    print(f"     ⚡ Gerando equity curve data para {nome_arquivo}")
                    equity_data = gerar_equity_curve_data(df, capital_inicial)
                    resultado_individual['equity_curve_data'] = equity_data
                
                # Processar trades individuais para este arquivo
                print(f"     📊 Processando trades para {nome_arquivo}")
                print(f"        📋 DataFrame shape: {df.shape}")
                print(f"        📅 Colunas disponíveis: {list(df.columns)}")
                trades_individual = processar_trades(df, {i: nome_arquivo})
                print(f"        ✅ Trades processados: {len(trades_individual)}")
                resultado_individual['trades'] = trades_individual
                
                resultado_individual['info_arquivo'] = {
                    "nome_arquivo": nome_arquivo,
                    "total_registros": len(df)
                }
                
                resultados_individuais[nome_arquivo] = make_json_serializable(resultado_individual)
                print(f"     ✅ Processado com sucesso: {nome_arquivo}")
                
            except Exception as e:
                print(f"❌ Erro ao processar arquivo {nome_arquivo}: {str(e)}")
                resultados_individuais[nome_arquivo] = {
                    "error": f"Erro ao processar arquivo: {str(e)}",
                    "info_arquivo": {
                        "nome_arquivo": nome_arquivo,
                        "total_registros": len(df)
                    }
                }
        
        print(f"📋 Resultados individuais processados: {list(resultados_individuais.keys())}")
        
        # Concatenar todos os DataFrames em um só para análise consolidada
        print(f"🔗 Processando dados consolidados:")
        print(f"   📊 Total de registros consolidados: {sum(len(df) for df in dataframes)}")
        
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        print(f"   📋 DataFrame consolidado criado com {len(df_consolidado)} registros")
        
        resultado_consolidado = processar_backtest_completo(df_consolidado, capital_inicial=capital_inicial, cdi=cdi)
        # Padronizar chaves também no consolidado
        try:
            if 'Day of Week Analysis' in resultado_consolidado:
                resultado_consolidado['day_of_week'] = resultado_consolidado['Day of Week Analysis']
            if 'Monthly Analysis' in resultado_consolidado:
                resultado_consolidado['monthly'] = resultado_consolidado['Monthly Analysis']
            if 'Equity Curve Data' in resultado_consolidado:
                resultado_consolidado['equity_curve_data'] = resultado_consolidado['Equity Curve Data']
        except Exception as e:
            print(f"⚠️ DEBUG: Falha ao padronizar chaves no consolidado: {e}")
        if 'equity_curve_data' not in resultado_consolidado:
            print(f"   ⚡ Gerando equity curve data consolidada")
            equity_data = gerar_equity_curve_data(df_consolidado, capital_inicial)
            resultado_consolidado['equity_curve_data'] = equity_data
        
        # Processar trades consolidados
        print(f"   📊 Processando trades consolidados")
        arquivo_para_indices = {}
        for i, nome_arquivo in enumerate(arquivos_processados):
            arquivo_para_indices[i] = nome_arquivo
        trades_consolidados = processar_trades(df_consolidado, arquivo_para_indices)
        resultado_consolidado['trades'] = trades_consolidados
        
        resultado_consolidado['info_arquivos'] = {
            "total_arquivos": len(arquivos_processados),
            "nomes_arquivos": arquivos_processados,
            "total_registros_consolidados": len(df_consolidado)
        }
        print(f"   ✅ Dados consolidados processados com sucesso")
        
        # Adicionar análises complementares ao consolidado
        if len(arquivos_processados) > 1:
            resultado_consolidado['day_of_week'] = calcular_day_of_week(df_consolidado)
            resultado_consolidado['monthly'] = calcular_monthly(df_consolidado)
        
        # Retornar estrutura com dados individuais e consolidados
        resultado_final = {
            "consolidado": make_json_serializable(resultado_consolidado),
            "individuais": resultados_individuais,
            "info_geral": {
                "total_arquivos": len(arquivos_processados),
                "nomes_arquivos": arquivos_processados,
                "modo_analise": "individual_e_consolidado"
            }
        }
        
        print(f"🎯 Resposta final preparada:")
        print(f"   📊 Arquivos individuais: {len(resultados_individuais)}")
        print(f"   🔗 Dados consolidados: ✅")
        print(f"   📋 Estrutura: {list(resultado_final.keys())}")
        
        return jsonify(resultado_final)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def gerar_equity_curve_data(df, capital_inicial=100000):
    """
    Função auxiliar para garantir que os dados da equity curve sejam gerados
    PADRONIZADO: Usa exatamente a mesma lógica do FunCalculos.py
    """
    try:
        # Encontrar coluna de resultado
        resultado_col = None
        data_col = None
        
        for col_name in ['operation_result', 'pnl', 'resultado']:
            if col_name in df.columns:
                resultado_col = col_name
                break
        
        for col_name in ['entry_date', 'data_abertura', 'data']:
            if col_name in df.columns:
                data_col = col_name
                break
        
        if resultado_col is None or data_col is None:
            return []
        
        # Filtrar dados válidos
        df_valid = df.dropna(subset=[resultado_col, data_col]).copy()
        
        if df_valid.empty:
            return []
        
        # Converter data se necessário
        if not pd.api.types.is_datetime64_any_dtype(df_valid[data_col]):
            df_valid[data_col] = pd.to_datetime(df_valid[data_col])
        
        # Ordenar por data
        df_valid = df_valid.sort_values(data_col).reset_index(drop=True)
        
        # PADRONIZADO: Usar exatamente a mesma lógica do FunCalculos.py
        # Calcular equity curve trade por trade (PADRONIZADO: apenas saldo cumulativo)
        df_valid['Saldo'] = df_valid[resultado_col].cumsum()
        df_valid['Saldo_Maximo'] = df_valid['Saldo'].cummax()
        df_valid['Drawdown'] = df_valid['Saldo'] - df_valid['Saldo_Maximo']
        
        # Calcular valor da carteira (para compatibilidade, mas não usado no drawdown)
        df_valid['Valor_Carteira'] = capital_inicial + df_valid['Saldo']
        df_valid['Peak_Carteira'] = capital_inicial + df_valid['Saldo_Maximo']
        
        # PADRONIZADO: Drawdown baseado apenas no saldo cumulativo (sem capital inicial)
        df_valid['Drawdown_Carteira'] = df_valid['Drawdown']  # Usar o mesmo drawdown do saldo
        df_valid['Drawdown_Percentual'] = (df_valid['Drawdown'] / df_valid['Saldo_Maximo'] * 100).fillna(0) if df_valid['Saldo_Maximo'].max() != 0 else 0
        
        # Preparar dados para o gráfico (igual ao FunCalculos.py)
        equity_curve = []
        
        # Ponto inicial
        equity_curve.append({
            "date": df_valid[data_col].iloc[0].strftime('%Y-%m-%d'),
            "fullDate": df_valid[data_col].iloc[0].strftime('%d/%m/%Y'),
            "saldo": 0.0,  # Saldo inicial sempre 0
            "valor": float(capital_inicial),  # Patrimônio inicial
            "resultado": 0.0,  # Resultado inicial sempre 0
            "drawdown": 0.0,
            "drawdownPercent": 0.0,
            "peak": float(capital_inicial),
            "trades": 0,
            "isStart": True
        })
        
        # Dados para cada trade (igual ao FunCalculos.py)
        for i, row in df_valid.iterrows():
            equity_curve.append({
                "date": row[data_col].strftime('%Y-%m-%d'),
                "fullDate": row[data_col].strftime('%d/%m/%Y %H:%M'),
                "saldo": float(row['Saldo']),  # ESTE é o valor que você quer mostrar
                "valor": float(row['Valor_Carteira']),  # Patrimônio total (saldo + capital)
                "resultado": float(row['Saldo']),  # Mantém compatibilidade
                "drawdown": float(abs(row['Drawdown_Carteira'])),  # Sempre positivo
                "drawdownPercent": float(abs(row['Drawdown_Percentual'])),
                "peak": float(row['Peak_Carteira']),
                "trades": int(i + 1),
                "trade_result": float(row[resultado_col]),  # Incluir mesmo se for 0
                "isStart": False
            })
        
        return equity_curve
        
    except Exception as e:
        print(f"Erro ao gerar equity curve data: {e}")
        return []

@app.route('/api/tabela', methods=['POST'])
def api_tabela():
    """
    Endpoint para processar arquivo único de backtest
    Suporta tanto arquivo único quanto múltiplos arquivos
    """
    print("🔍 DEBUG: api_tabela chamada!")
    print(f"🔍 DEBUG: request.files: {list(request.files.keys())}")
    print(f"🔍 DEBUG: request.form: {list(request.form.keys())}")
    
    try:
        # Lista para armazenar todos os DataFrames
        dataframes = []
        arquivos_processados = []
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            print(f"🔍 DEBUG: Arquivo recebido: {arquivo.filename}")
            print(f"🔍 DEBUG: Tipo do arquivo: {type(arquivo)}")
            if arquivo.filename != '':
                try:
                    df = carregar_csv_safe(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
                    print(f"🔍 DEBUG: Arquivo processado com sucesso")
                except Exception as e:
                    print(f"🔍 DEBUG: Erro ao processar arquivo: {e}")
                    raise e
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_safe(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Envie um arquivo ou caminho via POST"}), 400
        
        print(f"🔍 DEBUG: dataframes encontrados: {len(dataframes)}")
        for i, df in enumerate(dataframes):
            print(f"🔍 DEBUG: DataFrame {i}: shape={df.shape}, columns={df.columns.tolist()}")
        
        # Concatenar todos os DataFrames em um só
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # Usar processar_backtest_completo
        print(f"🔍 DEBUG: DataFrame shape: {df_consolidado.shape}")
        print(f"🔍 DEBUG: DataFrame columns: {df_consolidado.columns.tolist()}")
        print(f"🔍 DEBUG: Primeiras linhas: {df_consolidado.head()}")
        
        resultado = processar_backtest_completo(df_consolidado, capital_inicial=capital_inicial, cdi=cdi)
        
        print(f"🔍 DEBUG: Resultado keys: {resultado.keys()}")
        if 'Performance Metrics' in resultado:
            print(f"🔍 DEBUG: Performance Metrics: {resultado['Performance Metrics']}")
        else:
            print("🔍 DEBUG: Performance Metrics não encontrado")
        
        # Verificar se equity_curve_data existe, se não, gerar
        if 'equity_curve_data' not in resultado:
            equity_data = gerar_equity_curve_data(df_consolidado, capital_inicial)
            resultado['equity_curve_data'] = equity_data
        
        # Adicionar informações dos arquivos se múltiplos
        if len(arquivos_processados) > 1:
            resultado['info_arquivos'] = {
                "total_arquivos": len(arquivos_processados),
                "nomes_arquivos": arquivos_processados,
                "total_registros_consolidados": len(df_consolidado)
            }

        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ============ NOVA ROTA ESPECÍFICA PARA DADOS DO GRÁFICO ============

@app.route('/api/equity-curve', methods=['POST'])
def api_equity_curve():
    """Endpoint específico para dados da curva de equity"""
    try:
        if 'file' in request.files:
            df = carregar_csv_safe(request.files['file'])
        elif 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
        else:
            return jsonify({"error": "Envie um arquivo ou caminho via POST"}), 400

        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        tipo_agrupamento = request.form.get('tipo', 'daily')  # 'trade', 'daily', 'weekly', 'monthly'
        
        # Importar as funções específicas do gráfico
        from FunCalculos import calcular_dados_grafico, calcular_dados_grafico_agrupado
        
        # Calcular dados baseado no tipo solicitado
        if tipo_agrupamento == 'trade':
            dados = calcular_dados_grafico(df, capital_inicial)
        else:
            dados = calcular_dados_grafico_agrupado(df, capital_inicial, tipo_agrupamento)
        
        resultado = {
            "equity_curve_data": dados,
            "tipo": tipo_agrupamento,
            "capital_inicial": capital_inicial,
            "total_pontos": len(dados)
        }

        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ ROTA PARA BACKTEST COMPLETO ============

@app.route('/api/backtest-completo', methods=['POST'])
def api_backtest_completo():
    """Endpoint para backtest completo com todos os dados incluindo gráfico"""
    try:
        if 'file' in request.files:
            df = carregar_csv_safe(request.files['file'])
        elif 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
        else:
            return jsonify({"error": "Envie um arquivo ou caminho via POST"}), 400

        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # Usar a função completa
        resultado = processar_backtest_completo(df, capital_inicial=capital_inicial, cdi=cdi)
        
        # Adicionar metadados úteis
        resultado["metadata"] = {
            "total_trades": len(df),
            "capital_inicial": capital_inicial,
            "cdi": cdi,
            "periodo": {
                "inicio": df['entry_date'].min().isoformat() if not df.empty and 'entry_date' in df.columns else None,
                "fim": df['entry_date'].max().isoformat() if not df.empty and 'entry_date' in df.columns else None
            }
        }

        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ ROTA PARA TABELA MÚLTIPLA CORRIGIDA ============

@app.route('/api/correlacao', methods=['POST'])
def api_correlacao_data_direcao():
    try:
        arquivos_processados = []
        
        # Verificar se recebeu dados JSON
        if request.is_json:
            data = request.get_json()
            
            # Verificar se tem dados de arquivos no JSON
            if 'arquivo1' in data and 'arquivo2' in data:
                # Processar dados JSON (quando frontend envia dados já processados)
                try:
                    # Aqui você pode processar os dados JSON se necessário
                    # Por enquanto, vamos retornar um erro informativo
                    return jsonify({"error": "API de correlação espera arquivos CSV, não dados JSON. Use FormData com arquivos."}), 400
                except Exception as e:
                    return jsonify({"error": f"Erro ao processar dados JSON: {str(e)}"}), 500
        
        # Verificar se recebeu arquivos
        if 'files' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado. Envie arquivos CSV via FormData."}), 400
        
        files = request.files.getlist('files')
        
        if len(files) < 2:
            return jsonify({"error": "Precisa de pelo menos 2 arquivos"}), 400
        
        # Processar cada arquivo
        for file in files:
            try:
                df = carregar_csv_safe(file)  # Usar função com encoding seguro
                nome = file.filename.replace('.csv', '').replace('.xlsx', '')
                arquivos_processados.append({
                    'nome': nome,
                    'df': df
                })
            except Exception as e:
                return jsonify({"error": f"Erro ao processar {file.filename}: {str(e)}"}), 500
        
        # Calcular correlação por data e direção
        resultado = calcular_correlacao_por_data_e_direcao(arquivos_processados)
        
        return jsonify(make_json_serializable(resultado))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(silent=True) or {}
    # Debug leve: registrar headers e presença de chave (mascarando valor)
    try:
        hdr_key = request.headers.get('x-openai-key') or (request.headers.get('Authorization') or '').replace('Bearer ', '')
        masked = (hdr_key[:6] + '...' + hdr_key[-4:]) if hdr_key else None
        print(f"[/chat] headers received: x-openai-key present={bool(request.headers.get('x-openai-key'))}, auth_present={bool(request.headers.get('Authorization'))}, key={masked}")
        print(f"[/chat] body keys: {list(data.keys())}")
    except Exception:
        pass
    messages = data.get('messages', [])

    try:
        # SDK v1.x requer cliente explícito quando variável de ambiente não está carregada no processo
        # Prioridades para obter a chave: Header -> Authorization Bearer -> Body -> Env
        api_key = (
            request.headers.get('x-openai-key')
            or (request.headers.get('Authorization') or '').replace('Bearer ', '').strip() or None
            or (data.get('apiKey') if isinstance(data, dict) else None)
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            return jsonify({"error": "OPENAI_API_KEY não disponível. Envie no header 'x-openai-key' ou configure no backend."}), 500
        client = _OpenAIClient(api_key=api_key)
        # usar modelo disponível (gpt-4o-mini é mais acessível)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=False
        )
        # extrai role + content
        choice = resp.choices[0].message
        return jsonify({
            "role": choice.role,
            "content": choice.content
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ NOVAS ROTAS PARA TRADES ============

@app.route('/api/trades', methods=['POST'])
def api_trades():
    """Endpoint principal para análise de trades - suporta arquivo único ou múltiplos arquivos"""
    try:
        # Obter parâmetros opcionais
        taxa_corretagem = float(request.form.get('taxa_corretagem', 0.5))
        taxa_emolumentos = float(request.form.get('taxa_emolumentos', 0.03))
        
        # Lista para armazenar todos os DataFrames
        dataframes = []
        arquivos_processados = []
        arquivo_para_indices = {}  # Mapeamento de índice para nome do arquivo
        current_index = 0
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_trades(arquivo)
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
                
                # Mapear índices para este arquivo
                for i in range(len(df)):
                    arquivo_para_indices[current_index + i] = arquivo.filename
                current_index += len(df)
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_trades(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
                    
                    # Mapear índices para este arquivo
                    for i in range(len(df)):
                        arquivo_para_indices[current_index + i] = arquivo.filename
                    current_index += len(df)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_trades(path)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
            
            # Mapear índices para este arquivo
            for i in range(len(df)):
                arquivo_para_indices[current_index + i] = os.path.basename(path)
            current_index += len(df)
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo enviado. Use 'file' para um arquivo ou 'files' para múltiplos"}), 400
        
        # Concatenar todos os DataFrames em um só
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # Processar dados consolidados com mapeamento de arquivos
        trades = processar_trades(df_consolidado, arquivo_para_indices)
        estatisticas_gerais = calcular_estatisticas_gerais(df_consolidado)
        estatisticas_por_ativo = calcular_estatisticas_por_ativo(df_consolidado)
        estatisticas_temporais = calcular_estatisticas_temporais(df_consolidado)
        custos = calcular_custos_operacionais(df_consolidado, taxa_corretagem, taxa_emolumentos)
        
        # Extrair listas únicas para filtros
        available_assets = sorted([str(symbol) for symbol in df_consolidado['symbol'].unique() if pd.notna(symbol)])
        # Extrair estratégias únicas dos trades processados
        available_strategies = sorted(list(set([trade['strategy'] for trade in trades if trade['strategy']])))

        resultado = {
            "trades": trades,
            "statistics": {
                "general": estatisticas_gerais,
                "by_asset": estatisticas_por_ativo,
                "temporal": estatisticas_temporais,
                "costs": custos
            },
            "filters": {
                "available_assets": available_assets,
                "available_strategies": available_strategies
            },
            "metadata": {
                "total_records": len(df_consolidado),
                "valid_trades": len(trades),
                "date_range": {
                    "start": df_consolidado['entry_date'].min().isoformat() if df_consolidado['entry_date'].notna().any() else None,
                    "end": df_consolidado['entry_date'].max().isoformat() if df_consolidado['entry_date'].notna().any() else None
                },
                "info_arquivos": {
                    "total_arquivos": len(arquivos_processados),
                    "nomes_arquivos": arquivos_processados,
                    "total_registros_consolidados": len(df_consolidado)
                }
            }
        }

        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/trades/summary', methods=['POST'])
def api_trades_summary():
    """Endpoint para obter apenas um resumo das estatísticas"""
    try:
        # Carregar arquivo
        if 'file' in request.files:
            df = carregar_csv_trades(request.files['file'])
        elif 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_trades(path)
        else:
            return jsonify({"error": "Envie um arquivo ou caminho via POST"}), 400

        # Calcular apenas estatísticas essenciais
        estatisticas_gerais = calcular_estatisticas_gerais(df)
        custos = calcular_custos_operacionais(df)
        
        resultado = {
            "summary": estatisticas_gerais,
            "costs": custos,
            "total_records": len(df)
        }

        return jsonify(make_json_serializable(resultado))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============ NOVAS ROTAS PARA MÉTRICAS DIÁRIAS ============

@app.route('/api/trades/daily-metrics', methods=['POST'])
def api_daily_metrics():
    """Endpoint para obter métricas diárias usando FunCalculos.py"""
    try:
        # Carregar arquivo
        if 'file' in request.files:
            df = carregar_csv_trades(request.files['file'])
        elif 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 400
            df = carregar_csv_trades(path)
        else:
            return jsonify({"error": "Envie um arquivo ou caminho via POST"}), 400

        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # Usar FunCalculos.py para garantir consistência
        from FunCalculos import processar_backtest_completo
        
        # Processar backtest completo usando FunCalculos.py
        resultado = processar_backtest_completo(df, capital_inicial=capital_inicial, cdi=cdi)
        
        # Extrair apenas as métricas principais do resultado
        performance_metrics = resultado.get("Performance Metrics", {})
        
        # Converter para formato esperado pelo frontend
        metricas_principais = {
            "sharpe_ratio": performance_metrics.get("Sharpe Ratio", 0),
            "fator_recuperacao": performance_metrics.get("Recovery Factor", 0),
            "drawdown_maximo": -performance_metrics.get("Max Drawdown ($)", 0),  # Negativo para compatibilidade
            "drawdown_maximo_pct": performance_metrics.get("Max Drawdown (%)", 0),
            "drawdown_medio": performance_metrics.get("Average Drawdown ($)", 0),  # NOVO: DD Médio
            "dias_operados": performance_metrics.get("Active Days", 0),
            "resultado_liquido": performance_metrics.get("Net Profit", 0),
            "fator_lucro": performance_metrics.get("Profit Factor", 0),
            "win_rate": performance_metrics.get("Win Rate (%)", 0),
            "roi": (performance_metrics.get("Net Profit", 0) / capital_inicial * 100) if capital_inicial > 0 else 0,
            # Campos adicionais para compatibilidade
            "drawdown_maximo_padronizado": -performance_metrics.get("Max Drawdown ($)", 0),
            "drawdown_maximo_pct_padronizado": performance_metrics.get("Max Drawdown (%)", 0),
            "max_drawdown_padronizado": performance_metrics.get("Max Drawdown ($)", 0),
            "max_drawdown_pct_padronizado": performance_metrics.get("Max Drawdown (%)", 0),
            "capital_estimado": capital_inicial
        }
        
        # Estrutura de resposta compatível
        metricas = {
            "metricas_principais": metricas_principais,
            "ganhos_perdas": {
                "ganho_medio_diario": performance_metrics.get("Average Win", 0),
                "perda_media_diaria": performance_metrics.get("Average Loss", 0),
                "payoff_diario": performance_metrics.get("Payoff", 0),
                "ganho_maximo_diario": performance_metrics.get("Max Trade Gain", 0),
                "perda_maxima_diaria": abs(performance_metrics.get("Max Trade Loss", 0))
            },
            "estatisticas_operacao": {
                "media_operacoes_dia": performance_metrics.get("Avg Trades/Active Day", 0),
                "taxa_acerto_diaria": performance_metrics.get("Win Rate (%)", 0),
                "dias_vencedores_perdedores": "N/A",  # Não disponível no FunCalculos.py
                "dias_perdedores_consecutivos": performance_metrics.get("Max Consecutive Losses", 0),
                "dias_vencedores_consecutivos": performance_metrics.get("Max Consecutive Wins", 0)
            }
        }
        
        if not metricas:
            return jsonify({"error": "Não foi possível calcular métricas"}), 400
        
        return jsonify(make_json_serializable(metricas))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/trades/metrics-from-data', methods=['POST'])
def api_metrics_from_data():
    """Endpoint para calcular métricas a partir de dados JSON já processados"""
    try:
        print(f"🔍 DEBUG: Iniciando /api/trades/metrics-from-data")
        print(f"🔍 DEBUG: Content-Type: {request.content_type}")
        print(f"🔍 DEBUG: Content-Length: {request.content_length}")
        
        # Verificar se há dados no request
        if not request.data:
            print(f"❌ DEBUG: Request sem dados")
            return jsonify({"error": "Request sem dados"}), 400
        
        # Tentar obter JSON
        try:
            data = request.get_json()
            print(f"🔍 DEBUG: JSON parseado com sucesso")
        except Exception as json_error:
            print(f"❌ DEBUG: Erro ao fazer parse do JSON: {json_error}")
            print(f"🔍 DEBUG: Dados brutos: {request.data[:500]}...")
            return jsonify({"error": f"Erro ao fazer parse do JSON: {str(json_error)}"}), 400
        
        if not data:
            print(f"❌ DEBUG: Data é None após parse")
            return jsonify({"error": "Dados JSON inválidos"}), 400
        
        print(f"🔍 DEBUG: Chaves no data: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        if not isinstance(data, dict) or 'trades' not in data:
            print(f"❌ DEBUG: 'trades' não encontrado no data")
            return jsonify({"error": "Dados de trades não fornecidos"}), 400
        
        # Converter trades JSON para DataFrame
        trades_data = data['trades']
        
        if not trades_data:
            print(f"❌ DEBUG: Lista de trades vazia")
            return jsonify({"error": "Lista de trades vazia"}), 400
        
        print(f"🔍 DEBUG: Número de trades recebidos: {len(trades_data)}")
        
        # ✅ CORREÇÃO: Criar DataFrame com otimizações
        try:
            df = pd.DataFrame(trades_data)
            print(f"🔍 DEBUG: DataFrame criado com {len(df)} linhas e {len(df.columns)} colunas")
            print(f"🔍 DEBUG: Colunas: {list(df.columns)}")
        except Exception as df_error:
            print(f"❌ DEBUG: Erro ao criar DataFrame: {df_error}")
            return jsonify({"error": f"Erro ao criar DataFrame: {str(df_error)}"}), 400
        
        # ✅ CORREÇÃO: Converter datas com otimizações
        try:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            print(f"🔍 DEBUG: Datas convertidas com sucesso")
        except Exception as date_error:
            print(f"❌ DEBUG: Erro ao converter datas: {date_error}")
            return jsonify({"error": f"Erro ao converter datas: {str(date_error)}"}), 400
        
        # ✅ CORREÇÃO: Garantir que pnl seja numérico com otimizações
        try:
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
            print(f"🔍 DEBUG: PnL convertido para numérico")
        except Exception as pnl_error:
            print(f"❌ DEBUG: Erro ao converter PnL: {pnl_error}")
            return jsonify({"error": f"Erro ao converter PnL: {str(pnl_error)}"}), 400
        
        # ✅ CORREÇÃO: Parâmetros opcionais com valores padrão otimizados
        capital_inicial = float(data.get('capital_inicial', 100000))
        cdi = float(data.get('cdi', 0.12))
        
        print(f"🔍 DEBUG: Processando {len(df)} trades")
        print(f"🔍 DEBUG: Capital inicial: {capital_inicial}")
        print(f"🔍 DEBUG: CDI: {cdi}")
        
        # ✅ CORREÇÃO: Usar FunCalculos.py para garantir consistência com cache
        try:
            from FunCalculos import processar_backtest_completo
            print(f"🔍 DEBUG: FunCalculos importado com sucesso")
        except Exception as import_error:
            print(f"❌ DEBUG: Erro ao importar FunCalculos: {import_error}")
            return jsonify({"error": f"Erro ao importar FunCalculos: {str(import_error)}"}), 500
        
        # ✅ CORREÇÃO: Processar backtest completo usando FunCalculos.py com otimizações
        try:
            resultado = processar_backtest_completo(df, capital_inicial=capital_inicial, cdi=cdi)
            print(f"🔍 DEBUG: Backtest processado com sucesso")
        except Exception as backtest_error:
            print(f"❌ DEBUG: Erro ao processar backtest: {backtest_error}")
            return jsonify({"error": f"Erro ao processar backtest: {str(backtest_error)}"}), 500
        
        # ✅ CORREÇÃO: Extrair apenas as métricas principais do resultado com otimizações
        performance_metrics = resultado.get("Performance Metrics", {})
        
        print(f"🔍 DEBUG: Performance Metrics recebidas:")
        for key, value in performance_metrics.items():
            print(f"  {key}: {value}")
        
        # ✅ CORREÇÃO: Converter para formato esperado pelo frontend com otimizações
        metricas_principais = {
            "sharpe_ratio": performance_metrics.get("Sharpe Ratio", 0),
            "fator_recuperacao": performance_metrics.get("Recovery Factor", 0),
            "drawdown_maximo": -performance_metrics.get("Max Drawdown ($)", 0),  # Negativo para compatibilidade
            "drawdown_maximo_pct": performance_metrics.get("Max Drawdown (%)", 0),
            "drawdown_medio": performance_metrics.get("Average Drawdown ($)", 0),  # NOVO: DD Médio
            "dias_operados": performance_metrics.get("Active Days", 0),
            "resultado_liquido": performance_metrics.get("Net Profit", 0),
            "fator_lucro": performance_metrics.get("Profit Factor", 0),
            "win_rate": performance_metrics.get("Win Rate (%)", 0),
            "roi": (performance_metrics.get("Net Profit", 0) / capital_inicial * 100) if capital_inicial > 0 else 0,
            # Campos adicionais para compatibilidade
            "drawdown_maximo_padronizado": -performance_metrics.get("Max Drawdown ($)", 0),
            "drawdown_maximo_pct_padronizado": performance_metrics.get("Max Drawdown (%)", 0),
            "max_drawdown_padronizado": performance_metrics.get("Max Drawdown ($)", 0),
            "max_drawdown_pct_padronizado": performance_metrics.get("Max Drawdown (%)", 0),
            "capital_estimado": capital_inicial
        }
        
        print(f"🔍 DEBUG: Métricas principais mapeadas:")
        for key, value in metricas_principais.items():
            print(f"  {key}: {value}")
        
        # ✅ CORREÇÃO: Estrutura de resposta compatível com otimizações
        metricas = {
            "metricas_principais": metricas_principais,
            "ganhos_perdas": {
                "ganho_medio_diario": performance_metrics.get("Average Win", 0),
                "perda_media_diaria": performance_metrics.get("Average Loss", 0),
                "payoff_diario": performance_metrics.get("Payoff", 0),
                "ganho_maximo_diario": performance_metrics.get("Max Trade Gain", 0),
                "perda_maxima_diaria": abs(performance_metrics.get("Max Trade Loss", 0))
            },
            "estatisticas_operacao": {
                "media_operacoes_dia": performance_metrics.get("Avg Trades/Active Day", 0),
                "taxa_acerto_diaria": performance_metrics.get("Win Rate (%)", 0),
                "dias_vencedores_perdedores": "N/A",  # Não disponível no FunCalculos.py
                "dias_perdedores_consecutivos": performance_metrics.get("Max Consecutive Losses", 0),
                "dias_vencedores_consecutivos": performance_metrics.get("Max Consecutive Wins", 0)
            }
        }
        
        print(f"🔍 DEBUG: Resposta final preparada")
        print(f"🔍 DEBUG: DD Médio na resposta: {metricas['metricas_principais']['drawdown_medio']}")
        
        if not metricas:
            print(f"❌ DEBUG: Métricas vazias")
            return jsonify({"error": "Não foi possível calcular métricas"}), 400
        
        # ✅ CORREÇÃO: Tentar serializar a resposta com otimizações
        try:
            response_data = make_json_serializable(metricas)
            print(f"🔍 DEBUG: Resposta serializada com sucesso")
            return jsonify(response_data)
        except Exception as serialize_error:
            print(f"❌ DEBUG: Erro ao serializar resposta: {serialize_error}")
            return jsonify({"error": f"Erro ao serializar resposta: {str(serialize_error)}"}), 500

    except Exception as e:
        print(f"❌ Erro na API: {e}")
        import traceback
        print(f"❌ Traceback completo:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/position-sizing', methods=['POST'])
def api_position_sizing():
    """Endpoint específico para calcular métricas de position sizing"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nenhum arquivo selecionado"}), 400
        
        # Carregar CSV
        print(f"📊 Processando arquivo: {file.filename}")
        
        # Carregar CSV com headers corretos
        try:
            df = pd.read_csv(file, skiprows=5, sep=';', encoding='latin1', decimal=',', header=None)
            
            # Definir headers corretos
            expected_headers = [
                'Ativo', 'Abertura', 'Fechamento', 'Tempo Operação', 'Qtd Compra', 'Qtd Venda',
                'Lado', 'Preço Compra', 'Preço Venda', 'Preço de Mercado', 'Médio',
                'Res. Intervalo', 'Res. Intervalo (%)', 'Número Operação', 'Res. Operação', 'Res. Operação (%)',
                'Drawdown', 'Ganho Max.', 'Perda Max.', 'TET', 'Total'
            ]
            
            if len(df.columns) == len(expected_headers):
                df.columns = expected_headers
                print(f"📊 Headers atribuídos corretamente")
            else:
                print(f"⚠️ Número de colunas ({len(df.columns)}) não corresponde aos headers esperados ({len(expected_headers)})")
                return jsonify({"error": f"Formato de CSV inválido. Esperado {len(expected_headers)} colunas, encontrado {len(df.columns)}"}), 400
            
            # Processar datas com tratamento de NaT
            print(f"📊 Processando datas - DataFrame shape inicial: {df.shape}")
            
            if 'Abertura' in df.columns:
                print(f"📊 Processando coluna 'Abertura'")
                print(f"📊 Amostra de valores 'Abertura': {df['Abertura'].head(3).tolist()}")
                df['Abertura'] = pd.to_datetime(df['Abertura'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
                print(f"📊 Após conversão - valores NaT: {df['Abertura'].isna().sum()}")
                # Remover linhas com datas inválidas
                df_antes = len(df)
                df = df.dropna(subset=['Abertura'])
                df_depois = len(df)
                print(f"📊 Linhas removidas de 'Abertura': {df_antes - df_depois}")
                
            if 'Fechamento' in df.columns:
                print(f"📊 Processando coluna 'Fechamento'")
                df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')
                # Remover linhas com datas inválidas
                df_antes = len(df)
                df = df.dropna(subset=['Fechamento'])
                df_depois = len(df)
                print(f"📊 Linhas removidas de 'Fechamento': {df_antes - df_depois}")
            
            print(f"📊 DataFrame após processamento de datas: {df.shape}")
            
            # Limpar valores numéricos
            numeric_columns = ['Res. Operação', 'Res. Operação (%)', 'Preço Compra', 'Preço Venda', 
                              'Preço de Mercado', 'Médio', 'Res. Intervalo', 'Res. Intervalo (%)',
                              'Drawdown', 'Ganho Max.', 'Perda Max.', 'Qtd Compra', 'Qtd Venda']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric_value)
            
            # Renomear colunas
            column_mapping = {
                'Ativo': 'symbol', 'Abertura': 'entry_date', 'Fechamento': 'exit_date',
                'Tempo Operação': 'duration_str', 'Qtd Compra': 'qty_buy', 'Qtd Venda': 'qty_sell',
                'Lado': 'direction', 'Preço Compra': 'entry_price', 'Preço Venda': 'exit_price',
                'Preço de Mercado': 'market_price', 'Médio': 'avg_price', 'Res. Intervalo': 'pnl',
                'Res. Intervalo (%)': 'pnl_pct', 'Número Operação': 'trade_number',
                'Res. Operação': 'operation_result', 'Res. Operação (%)': 'operation_result_pct',
                'Drawdown': 'drawdown', 'Ganho Max.': 'max_gain', 'Perda Max.': 'max_loss',
                'TET': 'tet', 'Total': 'total'
            }
            df = df.rename(columns=column_mapping)
            
            # Converter direção
            if 'direction' in df.columns:
                df['direction'] = df['direction'].map({'C': 'long', 'V': 'short'}).fillna('long')
            
            # Usar operation_result como pnl
            if 'operation_result' in df.columns:
                df['pnl'] = df['operation_result']
            
            print(f"📊 DataFrame processado - Shape: {df.shape}, Colunas: {list(df.columns)}")
            
        except Exception as e:
            print(f"❌ Erro ao processar CSV: {e}")
            return jsonify({"error": f"Erro ao processar CSV: {e}"}), 400
        
        # Processar trades
        trades = processar_trades(df)
        print(f"📊 Trades processados: {len(trades)}")
        
        if not trades:
            print(f"❌ Nenhum trade válido encontrado")
            print(f"📊 DataFrame info:")
            print(f"   - Shape: {df.shape}")
            print(f"   - Colunas: {list(df.columns)}")
            print(f"   - Primeiras linhas:")
            if not df.empty:
                print(df.head(3).to_string())
            return jsonify({
                "error": "Nenhum trade válido encontrado",
                "debug": {
                    "dataframe_shape": df.shape,
                    "dataframe_columns": list(df.columns),
                    "sample_data": df.head(3).to_dict('records') if not df.empty else []
                }
            }), 400
        
        print(f"📊 Calculando position sizing para {len(trades)} trades")
        
        # Extrair dados de posição
        position_data = []
        for trade in trades:
            # Tentar diferentes campos de quantidade
            quantity = (trade.get('quantity_total', 0) or 
                       trade.get('quantity_compra', 0) or 
                       trade.get('quantity_venda', 0) or
                       trade.get('qty_buy', 0) or 
                       trade.get('qty_sell', 0) or 0)
            
            if quantity > 0:
                position_data.append({
                    'quantity': quantity,
                    'pnl': trade.get('pnl', 0),
                    'entry_price': trade.get('entry_price', 0),
                    'exit_price': trade.get('exit_price', 0)
                })
        
        print(f"📊 Dados de posição encontrados: {len(position_data)} trades com quantidade")
        
        if not position_data:
            return jsonify({
                "error": "Nenhum dado de posição encontrado nos trades",
                "available_fields": list(trades[0].keys()) if trades else []
            }), 400
        
        # Calcular estatísticas de posição
        quantities = [p['quantity'] for p in position_data]
        max_position = max(quantities) if quantities else 0
        avg_position = sum(quantities) / len(quantities) if quantities else 0
        median_position = sorted(quantities)[len(quantities)//2] if quantities else 0
        
        # Calcular risco por trade (baseado na perda média)
        losses = [abs(p['pnl']) for p in position_data if p['pnl'] < 0]
        avg_trade_risk = sum(losses) / len(losses) if losses else 0
        
        # Calcular account risk (2% do capital total)
        total_pnl = sum(t['pnl'] for t in trades)
        account_risk = max(0, total_pnl) * 0.02  # 2% rule
        
        # Calcular posição recomendada
        recommended_position = int(account_risk / avg_trade_risk) if avg_trade_risk > 0 else 0
        
        # Determinar tipo de ativo (ações vs futuros) com lógica melhorada
        avg_trade_value = abs(sum(t['pnl'] for t in trades) / len(trades))
        
        # Lógica melhorada para determinar se é ações ou futuros
        # Se tem posições > 100 ou trade value > 1000, provavelmente é ações
        is_stocks = avg_position > 100 or avg_trade_value > 1000
        
        # Se não tem dados de posição, usar trade value como critério
        if avg_position == 0:
            is_stocks = avg_trade_value > 500  # Se trade value > 500, provavelmente ações
        
        # Calcular dados para AMBOS os tipos de ativo (sempre)
        # Para Ações - usar dados reais ou estimar baseado no trade value
        stocks_avg_position = avg_position if is_stocks else max(1, int(avg_trade_value * 10))  # Estimativa para ações
        stocks_max_position = max_position if is_stocks else stocks_avg_position * 2
        stocks_median_position = median_position if is_stocks else stocks_avg_position
        stocks_recommended = recommended_position if is_stocks else max(1, int(account_risk / (avg_trade_risk * 10)))  # Ações têm menor risco
        
        print(f"📊 Análise de tipo de ativo:")
        print(f"   - Posição média: {avg_position}")
        print(f"   - Trade value médio: {avg_trade_value}")
        print(f"   - Tipo determinado: {'Ações' if is_stocks else 'Futuros'}")
        print(f"📊 Cálculos para Ações:")
        print(f"   - Posição média estimada: {stocks_avg_position}")
        print(f"   - Posição máxima: {stocks_max_position}")
        print(f"   - Posição recomendada: {stocks_recommended}")
        print(f"📊 Cálculos para Futuros:")
        print(f"   - Posição média real: {avg_position}")
        print(f"   - Posição máxima: {max_position}")
        print(f"   - Posição recomendada: {recommended_position}")
        
        # Calcular posições abertas máximas
        trades_by_date = {}
        for trade in trades:
            # Usar entry_date que já foi renomeado de 'Abertura'
            entry_date = trade.get('entry_date', '')
            if entry_date:
                date = entry_date[:10]  # YYYY-MM-DD
                if date not in trades_by_date:
                    trades_by_date[date] = []
                trades_by_date[date].append(trade)
        
        max_open_positions = max(len(trades) for trades in trades_by_date.values()) if trades_by_date else 0
        
        stocks_data = {
            "maxPositionPerTrade": stocks_max_position,
            "avgPositionPerTrade": round(stocks_avg_position),
            "medianPositionPerTrade": stocks_median_position,
            "avgLeverage": 0.85,
            "recommendedPosition": stocks_recommended,
            "riskPerTrade": round(avg_trade_risk * 10, 2)  # Ações têm risco por trade maior
        }
        
        # Para Futuros - usar dados reais
        futures_data = {
            "maxPositionPerTrade": max_position if not is_stocks else max_position,
            "avgPositionPerTrade": round(avg_position) if not is_stocks else avg_position,
            "medianPositionPerTrade": median_position if not is_stocks else median_position,
            "avgLeverage": 3.2,
            "recommendedPosition": recommended_position if not is_stocks else recommended_position,
            "riskPerTrade": round(avg_trade_risk, 2)
        }
        
        # Se não há dados de posição, estimar para ambos
        if avg_position == 0:
            # Estimar posição baseada no trade value
            estimated_position = max(1, int(avg_trade_value / 100))
            
            # Para ações - estimativa mais conservadora
            stocks_estimated = max(1, int(avg_trade_value * 5))
            stocks_data.update({
                "maxPositionPerTrade": stocks_estimated * 2,
                "avgPositionPerTrade": stocks_estimated,
                "medianPositionPerTrade": stocks_estimated,
                "recommendedPosition": max(1, int(account_risk / (avg_trade_risk * 5)))
            })
            
            # Para futuros - estimativa baseada no trade value
            futures_data.update({
                "maxPositionPerTrade": estimated_position * 2,
                "avgPositionPerTrade": estimated_position,
                "medianPositionPerTrade": estimated_position,
                "recommendedPosition": estimated_position
            })
        
        result = {
            "stocks": stocks_data,
            "futures": futures_data,
            "general": {
                "maxOpenPositions": max_open_positions,
                "setupsMaximosPorDia": max_open_positions,
                "accountRisk": round(account_risk, 2),
                "maxRiskPerTrade": round(account_risk * 0.5, 2)  # 1% rule
            },
            "debug": {
                "totalTrades": len(trades),
                "tradesWithPosition": len(position_data),
                "assetType": "Stocks" if is_stocks else "Futures",
                "avgTradeValue": round(avg_trade_value, 2),
                "avgPosition": round(avg_position, 2),
                "isStocks": is_stocks,
                "hasPositionData": len(position_data) > 0
            }
        }
        
        print(f"📊 Position sizing calculado: {result}")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erro em api_position_sizing: {e}")
        return jsonify({"error": str(e)}), 500

def debug_drawdown_calculation(df: pd.DataFrame) -> Dict[str, float]:
    """
    Função de debug para verificar se todos os cálculos de drawdown estão padronizados
    """
    if df.empty:
        return {}
    
    print("🔍 DEBUG - Verificação de padronização do drawdown:")
    
    # Método 1: FunCalculos.py (trades individuais)
    df_valid = df.dropna(subset=['pnl', 'entry_date']).copy()
    df_valid = df_valid.sort_values('entry_date').reset_index(drop=True)
    
    equity = df_valid['pnl'].cumsum()
    peak = equity.cummax()
    dd_ser = equity - peak
    max_dd_funcalculos = abs(dd_ser.min()) if not dd_ser.empty else 0
    pct_dd_funcalculos = (max_dd_funcalculos / equity.iloc[-1] * 100) if equity.iloc[-1] != 0 else 0
    
    print(f"  FunCalculos.py: R$ {max_dd_funcalculos:.2f} ({pct_dd_funcalculos:.2f}%)")
    
    # Método 2: Análise diária (dias consolidados)
    df_valid['date'] = pd.to_datetime(df_valid['entry_date']).dt.date
    daily_stats = df_valid.groupby('date').agg({
        'pnl': ['sum', 'count', 'mean'],
    }).round(2)
    
    daily_stats.columns = ['total_pnl', 'total_trades', 'avg_pnl']
    daily_stats['cumulative_pnl'] = daily_stats['total_pnl'].cumsum()
    daily_stats['running_max'] = daily_stats['cumulative_pnl'].expanding().max()
    daily_stats['drawdown'] = daily_stats['cumulative_pnl'] - daily_stats['running_max']
    
    max_dd_daily = abs(daily_stats['drawdown'].min()) if not daily_stats['drawdown'].empty else 0
    pct_dd_daily = (max_dd_daily / daily_stats['cumulative_pnl'].iloc[-1] * 100) if daily_stats['cumulative_pnl'].iloc[-1] != 0 else 0
    
    print(f"  Análise Diária: R$ {max_dd_daily:.2f} ({pct_dd_daily:.2f}%)")
    
    # Método 3: Gráfico (calcular_dados_grafico)
    grafico_data = calcular_dados_grafico(df_valid)
    if grafico_data:
        drawdowns_grafico = [abs(item['drawdown']) for item in grafico_data if not item.get('isStart', False)]
        max_dd_grafico = max(drawdowns_grafico) if drawdowns_grafico else 0
        print(f"  Gráfico: R$ {max_dd_grafico:.2f}")
    else:
        print(f"  Gráfico: N/A")
    
    # Verificar se todos os métodos produzem o mesmo resultado
    methods = [
        ("FunCalculos.py", max_dd_funcalculos),
        ("Análise Diária", max_dd_daily),
        ("Gráfico", max_dd_grafico if 'max_dd_grafico' in locals() else 0)
    ]
    
    all_equal = len(set(method[1] for method in methods)) == 1
    print(f"  ✅ Todos os métodos iguais: {all_equal}")
    
    if not all_equal:
        print("  ⚠️ DIFERENÇAS ENCONTRADAS:")
        for method_name, value in methods:
            print(f"    {method_name}: R$ {value:.2f}")
    
    return {
        "funcalculos": max_dd_funcalculos,
        "daily": max_dd_daily,
        "grafico": max_dd_grafico if 'max_dd_grafico' in locals() else 0,
        "all_equal": all_equal
    }

def calcular_drawdown_padronizado(df: pd.DataFrame) -> Dict[str, float]:
    """
    Função centralizada para calcular drawdown de forma padronizada
    Usada em todas as seções para garantir consistência
    """
    if df.empty:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "saldo_final": 0.0,
            "capital_inicial": 0.0
        }
    
    # Filtrar trades válidas
    df_valid = df.dropna(subset=['pnl', 'entry_date']).copy()
    df_valid = df_valid.sort_values('entry_date').reset_index(drop=True)
    
    if df_valid.empty:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "saldo_final": 0.0,
            "capital_inicial": 0.0
        }
    
    # Calcular equity curve trade por trade (PADRONIZADO)
    df_valid['equity'] = df_valid['pnl'].cumsum()
    df_valid['peak'] = df_valid['equity'].cummax()
    df_valid['drawdown'] = df_valid['equity'] - df_valid['peak']
    
    # Drawdown máximo (valor positivo)
    max_drawdown = abs(df_valid['drawdown'].min()) if not df_valid['drawdown'].empty else 0.0
    
    # Saldo final
    saldo_final = df_valid['equity'].iloc[-1] if not df_valid['equity'].empty else 0.0
    
    # Capital inicial estimado (baseado no pico máximo)
    capital_inicial = df_valid['peak'].max() if not df_valid['peak'].empty else 0.0
    
    # Percentual do drawdown (baseado no capital inicial)
    max_drawdown_pct = (max_drawdown / capital_inicial * 100) if capital_inicial != 0 else 0.0
    
    # Logs de debug
    print(f"🔍 DEBUG - Drawdown Padronizado:")
    print(f"  Max Drawdown: R$ {max_drawdown:.2f}")
    print(f"  Max Drawdown %: {max_drawdown_pct:.2f}%")
    print(f"  Saldo Final: R$ {saldo_final:.2f}")
    print(f"  Capital Inicial: R$ {capital_inicial:.2f}")
    
    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "saldo_final": saldo_final,
        "capital_inicial": capital_inicial
    }

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0',
                port=5002,
                debug=False,
                use_reloader=False)
    except Exception as e:
        print(f"Erro ao iniciar servidor: {e}")
        import traceback
        traceback.print_exc()