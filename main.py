from flask import Flask, request, jsonify
import os
from flask_cors import CORS
import openai
from openai import OpenAI as _OpenAIClient
from FunMultiCalculos import processar_multiplos_arquivos, processar_multiplos_arquivos_comparativo
from Correlacao import *
from FunCalculos import carregar_csv, calcular_performance, calcular_day_of_week, calcular_monthly, processar_backtest_completo, calcular_dados_grafico, _normalize_trades_dataframe
import dotenv
import os.path as _path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from functools import wraps
import jwt
from supabase import create_client, Client

# Carregar variáveis de ambiente de múltiplas localizações para maior robustez
# Tenta carregar de vários locais possíveis
env_loaded = False
base_paths = [
    _path.dirname(__file__),  # devhubback/
    _path.join(_path.dirname(__file__), '..'),  # python-freela/
    os.getcwd(),  # diretório atual de execução
]

# Primeiro tenta carregar .env, depois tenta .env.backup.*
for base_path in base_paths:
    env_path = _path.join(base_path, '.env')
    if _path.exists(env_path):
        result = dotenv.load_dotenv(dotenv_path=env_path, override=False)
        if result:
            print(f"[INFO] Arquivo .env carregado de: {_path.abspath(env_path)}")
            env_loaded = True
            break

# Se não encontrou .env, tenta arquivos de backup
if not env_loaded:
    import glob
    for base_path in base_paths:
        backup_pattern = _path.join(base_path, '.env.backup.*')
        backup_files = glob.glob(backup_pattern)
        if backup_files:
            # Pega o mais recente
            backup_files.sort(reverse=True)
            result = dotenv.load_dotenv(dotenv_path=backup_files[0], override=False)
            if result:
                print(f"[INFO] Arquivo .env carregado de backup: {_path.abspath(backup_files[0])}")
                env_loaded = True
                break

# Se ainda não encontrou, tenta o padrão do python-dotenv
if not env_loaded:
    result = dotenv.load_dotenv()
    if result:
        print(f"[INFO] Arquivo .env carregado do diretório atual: {os.getcwd()}")
        env_loaded = True

if not env_loaded:
    print("[WARN] Nenhum arquivo .env encontrado. Verifique se o arquivo existe em:")
    for base_path in base_paths:
        print(f"  - {_path.abspath(_path.join(base_path, '.env'))}")

# main.py
app = Flask(__name__)

# Configuração CORS para permitir acesso do frontend
CORS(app, 
     resources={r"/api/*": {
         "origins": [
             'http://localhost:4173',
             'http://localhost:5173',
             'http://localhost:5174',
             'http://localhost:3000',
             'https://devhubtrader.com.br',
             'https://www.devhubtrader.com.br',
             'http://devhubtrader.com.br',
             'http://www.devhubtrader.com.br'
         ],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
         "allow_headers": ["Content-Type", "Authorization", "x-openai-key", "X-Requested-With"],
         "supports_credentials": True,
         "max_age": 3600
     }},
     supports_credentials=True)

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

# ============ CONFIGURAÇÃO SUPABASE ============
# Debug: verificar variáveis relacionadas ao Supabase no ambiente
supabase_vars = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL"),
    "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),
    "VITE_SUPABASE_URL": os.getenv("VITE_SUPABASE_URL"),  # Pode estar com prefixo VITE_
    "VITE_SUPABASE_ANON_KEY": os.getenv("VITE_SUPABASE_ANON_KEY"),
}
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# PRIORIDADE 1: SUPABASE_URL (variável padrão)
SUPABASE_URL = supabase_vars["SUPABASE_URL"]
# Se não encontrou, tenta usar VITE_SUPABASE_URL
if not SUPABASE_URL and supabase_vars["VITE_SUPABASE_URL"]:
    SUPABASE_URL = supabase_vars["VITE_SUPABASE_URL"]
    print(f"[INFO] Usando VITE_SUPABASE_URL como SUPABASE_URL")

# Limpar e validar URL
if SUPABASE_URL:
    SUPABASE_URL = SUPABASE_URL.strip()
    # Validar formato básico da URL
    if not SUPABASE_URL.startswith('http://') and not SUPABASE_URL.startswith('https://'):
        print(f"[ERROR] SUPABASE_URL inválida (deve começar com http:// ou https://): {SUPABASE_URL[:50]}...")
        SUPABASE_URL = None

# PRIORIDADE: Sempre usar SERVICE_ROLE_KEY primeiro (bypassa RLS)
# Se não tiver, usar ANON_KEY como fallback (com aviso)
SUPABASE_KEY = None
if supabase_vars["SUPABASE_SERVICE_ROLE_KEY"]:
    SUPABASE_KEY = supabase_vars["SUPABASE_SERVICE_ROLE_KEY"].strip()
    print(f"[INFO] ✅ Usando SUPABASE_SERVICE_ROLE_KEY (bypassa RLS)")
elif supabase_vars["SUPABASE_ANON_KEY"]:
    SUPABASE_KEY = supabase_vars["SUPABASE_ANON_KEY"].strip()
    print(f"[WARN] ⚠️  Usando SUPABASE_ANON_KEY - operações podem falhar por RLS!")
    print(f"[WARN] Configure SUPABASE_SERVICE_ROLE_KEY no .env para bypassar RLS")
elif supabase_vars["VITE_SUPABASE_ANON_KEY"]:
    SUPABASE_KEY = supabase_vars["VITE_SUPABASE_ANON_KEY"].strip()
    print(f"[WARN] ⚠️  Usando VITE_SUPABASE_ANON_KEY - operações podem falhar por RLS!")
    print(f"[WARN] Configure SUPABASE_SERVICE_ROLE_KEY no .env para bypassar RLS")

supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        # Debug: mostrar URL e tamanho da chave (sem mostrar a chave completa)
        print(f"[DEBUG] Tentando conectar ao Supabase...")
        print(f"[DEBUG] URL: {SUPABASE_URL}")
        print(f"[DEBUG] Chave configurada: {'✓' if SUPABASE_KEY else '✗'} (tamanho: {len(SUPABASE_KEY) if SUPABASE_KEY else 0} caracteres)")
        
        # Verificar qual tipo de chave está sendo usada
        is_service_role = bool(supabase_vars["SUPABASE_SERVICE_ROLE_KEY"])
        if is_service_role:
            print("[INFO] Usando SUPABASE_SERVICE_ROLE_KEY (bypassa RLS)")
        else:
            print("[WARN] ⚠️  USANDO ANON_KEY - operações podem falhar por RLS!")
            print("[WARN] Configure SUPABASE_SERVICE_ROLE_KEY no .env")
        
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        if is_service_role:
            print("[INFO] ✅ Cliente Supabase inicializado com SERVICE_ROLE_KEY (bypassa RLS)")
        else:
            print("[WARN] ⚠️  Cliente Supabase inicializado com ANON_KEY (respeita RLS)")
            print("[WARN] Operações administrativas podem falhar. Use SERVICE_ROLE_KEY.")
    except Exception as e:
        print(f"[WARN] Erro ao inicializar Supabase: {e}")
        print(f"[DEBUG] SUPABASE_URL recebida: '{SUPABASE_URL}' (tipo: {type(SUPABASE_URL)}, tamanho: {len(SUPABASE_URL) if SUPABASE_URL else 0})")
        print(f"[DEBUG] SUPABASE_KEY recebida: {'✓' if SUPABASE_KEY else '✗'} (tamanho: {len(SUPABASE_KEY) if SUPABASE_KEY else 0})")
        import traceback
        traceback.print_exc()
        print("[WARN] Continuando sem Supabase - rotas de usuário não funcionarão")
else:
    print("[WARN] Variáveis SUPABASE_URL ou SUPABASE_KEY não encontradas.")
    print(f"[DEBUG] SUPABASE_URL: {'✓' if SUPABASE_URL else '✗'}, SUPABASE_KEY: {'✓' if SUPABASE_KEY else '✗'}")
    print("[DEBUG] Variáveis encontradas no ambiente:")
    for var_name, var_value in supabase_vars.items():
        print(f"  {var_name}: {'✓ (definida)' if var_value else '✗ (não encontrada)'}")
    print("[WARN] Rotas de usuário não funcionarão.")

# ============ FUNÇÕES HELPER PARA AUTENTICAÇÃO ============
def get_user_id_from_token() -> Optional[str]:
    """
    Extrai o user_id do token JWT do Supabase no header Authorization
    Retorna None se não conseguir autenticar
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    try:
        # Formato: "Bearer <token>"
        token = auth_header.replace('Bearer ', '').strip()
        if not token:
            return None
        
        # Se temos JWT_SECRET, validar o token
        if SUPABASE_JWT_SECRET:
            try:
                decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=['HS256'], options={"verify_signature": True})
                return decoded.get('sub')  # 'sub' é o user_id no JWT do Supabase
            except jwt.ExpiredSignatureError:
                return None
            except jwt.InvalidTokenError:
                return None
        
        # Se não temos JWT_SECRET, tentar decodificar sem validação (apenas para desenvolvimento)
        # Em produção, sempre use JWT_SECRET
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            return decoded.get('sub')
        except:
            return None
            
    except Exception as e:
        print(f"[ERROR] Erro ao decodificar token: {e}")
        return None

def require_auth(f):
    """Decorator para rotas que requerem autenticação"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = get_user_id_from_token()
        if not user_id:
            return jsonify({"error": "Não autenticado. Token inválido ou ausente."}), 401
        # Adicionar user_id ao request para uso na função
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated_function

def get_user_supabase_client() -> Optional[Client]:
    """
    Retorna um cliente Supabase para operações com RLS.
    Se SERVICE_ROLE_KEY estiver configurado, usa o cliente global (bypassa RLS).
    Caso contrário, cria um cliente com ANON_KEY e passa o token do usuário nos headers.
    """
    # Se estamos usando SERVICE_ROLE_KEY, o cliente global já bypassa RLS
    if supabase_vars.get("SUPABASE_SERVICE_ROLE_KEY"):
        return supabase_client
    
    # Se não temos SERVICE_ROLE_KEY, precisamos usar o token do usuário
    if not SUPABASE_URL:
        return None
    
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return None
    
    try:
        # Extrair o token
        token = auth_header.replace('Bearer ', '').strip()
        if not token:
            return None
        
        # Usar ANON_KEY para criar cliente
        anon_key = supabase_vars.get("SUPABASE_ANON_KEY") or supabase_vars.get("VITE_SUPABASE_ANON_KEY")
        if not anon_key:
            print("[WARN] ANON_KEY não encontrada. Usando cliente global (pode falhar com RLS).")
            return supabase_client
        
        # Criar cliente com ANON_KEY
        # O token será passado nos headers das requisições
        user_client = create_client(SUPABASE_URL, anon_key)
        
        # Armazenar o token para uso nas requisições
        # Nota: O cliente Supabase Python não suporta set_session diretamente,
        # mas podemos passar o token nos headers manualmente se necessário
        return user_client
    except Exception as e:
        print(f"[WARN] Erro ao criar cliente Supabase com token do usuário: {e}")
        return supabase_client

# ============ MIDDLEWARE PARA LOG E CORS ============
@app.before_request
def log_request_info():
    """Log das requisições para debug"""
    # Silent request logging
    pass

@app.after_request
def after_request(response):
    """Adiciona headers CORS em todas as respostas"""
    origin = request.headers.get('Origin')
    allowed_origins = [
        'http://localhost:4173',
        'http://localhost:5173',
        'http://localhost:5174',
        'http://localhost:3000',
        'https://devhubtrader.com.br',
        'https://www.devhubtrader.com.br',
        'http://devhubtrader.com.br',
        'http://www.devhubtrader.com.br'
    ]
    
    # Sempre adiciona headers CORS se a origem for permitida
    if origin and origin in allowed_origins:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    # Sempre adiciona métodos e headers permitidos
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, x-openai-key, X-Requested-With'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '3600'
    
    return response

@app.before_request
def handle_preflight():
    """Trata requisições OPTIONS (preflight)"""
    if request.method == "OPTIONS":
        response = jsonify({})
        origin = request.headers.get('Origin')
        allowed_origins = [
            'http://localhost:4173',
            'http://localhost:5173',
            'http://localhost:5174',
            'http://localhost:3000',
            'https://devhubtrader.com.br',
            'https://www.devhubtrader.com.br',
            'http://devhubtrader.com.br',
            'http://www.devhubtrader.com.br'
        ]
        if origin and origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, x-openai-key, X-Requested-With'
        response.headers['Access-Control-Max-Age'] = '3600'
        return response

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


def _parse_filters_from_request(req) -> Dict[str, Any]:
    """Extrai filtros básicos do request (direção, datas, etc.)."""
    filters: Dict[str, Any] = {}

    # CORREÇÃO: Verificar se os filtros vêm via JSON no body
    if req.is_json:
        try:
            json_data = req.get_json(silent=True)
            if json_data and isinstance(json_data, dict):
                # Se há um campo 'filters' no JSON, usar ele
                if 'filters' in json_data:
                    filters.update(json_data['filters'])
                # Ou se os filtros estão diretamente no JSON
                else:
                    # Extrair filtros diretamente do JSON
                    for key in ('direction', 'direcao', 'directions', 'side', 'date_from', 'date_to', 
                               'data_inicio', 'data_fim', 'asset', 'symbol', 'ativo', 'simbolo', 
                               'strategy', 'estrategia'):
                        if key in json_data:
                            filters[key] = json_data[key]
        except Exception as e:
            print(f"⚠️ Erro ao processar filtros do JSON: {e}")

    # Extrair filtros do form (FormData)
    raw_filters = req.form.get('filters') or req.form.get('filtros')
    if raw_filters:
        try:
            parsed = json.loads(raw_filters)
            if isinstance(parsed, dict):
                filters.update(parsed)
        except json.JSONDecodeError:
            pass

    # Extrair direção do form
    for key in ('direction', 'direcao', 'directions', 'side'):
        value = req.form.get(key)
        if value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            filters['direction'] = parsed
            break

    # Extrair filtros de data diretamente do request (form ou JSON)
    date_from = (filters.get('date_from') or 
                 req.form.get('date_from') or 
                 req.form.get('data_inicio') or 
                 req.form.get('date_start') or
                 (req.get_json(silent=True) or {}).get('date_from') or
                 (req.get_json(silent=True) or {}).get('data_inicio'))
    
    date_to = (filters.get('date_to') or 
               req.form.get('date_to') or 
               req.form.get('data_fim') or 
               req.form.get('date_end') or
               (req.get_json(silent=True) or {}).get('date_to') or
               (req.get_json(silent=True) or {}).get('data_fim'))
    
    if date_from:
        filters['date_from'] = date_from
    if date_to:
        filters['date_to'] = date_to

    # Extrair outros filtros diretamente do request (form ou JSON)
    for key in ('asset', 'symbol', 'ativo', 'simbolo', 'strategy', 'estrategia'):
        value = (filters.get(key) or 
                req.form.get(key) or
                (req.get_json(silent=True) or {}).get(key))
        if value:
            filters[key] = value

    # CORREÇÃO: Extrair filtros de dia da semana (day_of_week, dia_semana, dayOfWeek)
    day_of_week_filter = (
        filters.get('day_of_week') or 
        filters.get('dia_semana') or 
        filters.get('dayOfWeek') or
        req.form.get('day_of_week') or 
        req.form.get('dia_semana') or
        req.form.get('dayOfWeek') or
        (req.get_json(silent=True) or {}).get('day_of_week') or
        (req.get_json(silent=True) or {}).get('dia_semana') or
        (req.get_json(silent=True) or {}).get('dayOfWeek')
    )
    if day_of_week_filter and day_of_week_filter not in ('Todos', 'All', 'todos', 'all', ''):
        filters['day_of_week'] = day_of_week_filter

    # CORREÇÃO: Extrair filtros de mês (month, mes, month_filter)
    month_filter = (
        filters.get('month') or 
        filters.get('mes') or 
        filters.get('month_filter') or
        req.form.get('month') or 
        req.form.get('mes') or
        req.form.get('month_filter') or
        (req.get_json(silent=True) or {}).get('month') or
        (req.get_json(silent=True) or {}).get('mes') or
        (req.get_json(silent=True) or {}).get('month_filter')
    )
    if month_filter and month_filter not in ('Todos', 'All', 'todos', 'all', ''):
        filters['month'] = month_filter

    # CORREÇÃO: Extrair filtros de horário (time_from, time_to, time_range, hora_inicio, hora_fim)
    # Também suporta faixas pré-definidas (abertura, meio_dia, tarde, pos_mercado)
    time_range = (
        filters.get('time_range') or 
        filters.get('faixa_horario') or 
        filters.get('predefined_range') or
        req.form.get('time_range') or 
        req.form.get('faixa_horario') or
        req.form.get('predefined_range') or
        (req.get_json(silent=True) or {}).get('time_range') or
        (req.get_json(silent=True) or {}).get('faixa_horario') or
        (req.get_json(silent=True) or {}).get('predefined_range')
    )
    
    # Mapear faixas pré-definidas para horários
    predefined_ranges = {
        'abertura': ('09:00', '11:00'),
        'opening': ('09:00', '11:00'),
        'meio_dia': ('11:00', '14:00'),
        'mid_day': ('11:00', '14:00'),
        'meio-dia': ('11:00', '14:00'),
        'tarde': ('14:00', '17:30'),
        'afternoon': ('14:00', '17:30'),
        'pos_mercado': ('17:30', '21:00'),
        'after_market': ('17:30', '21:00'),
        'pós-mercado': ('17:30', '21:00'),
        'pos-mercado': ('17:30', '21:00')
    }
    
    # Se tem faixa pré-definida, usar ela
    if time_range and time_range.lower() in predefined_ranges:
        time_from, time_to = predefined_ranges[time_range.lower()]
        filters['time_from'] = time_from
        filters['time_to'] = time_to
    else:
        # Caso contrário, usar horários customizados
        time_from = (
            filters.get('time_from') or 
            filters.get('hora_inicio') or 
            filters.get('time_start') or
            req.form.get('time_from') or 
            req.form.get('hora_inicio') or
            req.form.get('time_start') or
            (req.get_json(silent=True) or {}).get('time_from') or
            (req.get_json(silent=True) or {}).get('hora_inicio') or
            (req.get_json(silent=True) or {}).get('time_start')
        )
        if time_from:
            filters['time_from'] = time_from

        time_to = (
            filters.get('time_to') or 
            filters.get('hora_fim') or 
            filters.get('time_end') or
            req.form.get('time_to') or 
            req.form.get('hora_fim') or
            req.form.get('time_end') or
            (req.get_json(silent=True) or {}).get('time_to') or
            (req.get_json(silent=True) or {}).get('hora_fim') or
            (req.get_json(silent=True) or {}).get('time_end')
        )
        if time_to:
            filters['time_to'] = time_to

    # CORREÇÃO: Extrair data específica (specific_date, data_especifica, specificDate)
    specific_date = (
        filters.get('specific_date') or 
        filters.get('data_especifica') or 
        filters.get('specificDate') or
        req.form.get('specific_date') or 
        req.form.get('data_especifica') or
        req.form.get('specificDate') or
        (req.get_json(silent=True) or {}).get('specific_date') or
        (req.get_json(silent=True) or {}).get('data_especifica') or
        (req.get_json(silent=True) or {}).get('specificDate')
    )
    if specific_date:
        filters['specific_date'] = specific_date

    # Log dos filtros extraídos para debug
    if filters:
        print(f"🔍 Filtros extraídos do request: {filters}")

    return filters


_DIRECTION_MAP = {
    'long': {'long'},
    'short': {'short'},
    'buy': {'long'},
    'sell': {'short'},
    'compra': {'long'},
    'venda': {'short'},
    'comprado': {'long'},
    'vendido': {'short'},
    'c': {'long'},
    'v': {'short'},
    'compra+venda': {'long', 'short'},
    'compra + venda': {'long', 'short'},
    'all': set(),
    'todos': set(),
    'ambos': {'long', 'short'},
    'ambas': {'long', 'short'}
}


def aplicar_filtros_basicos(df: pd.DataFrame, filtros: Dict[str, Any]) -> pd.DataFrame:
    """Aplica filtros padrão (direção, ativo, estratégia, datas, etc.) ao DataFrame."""
    # CORREÇÃO: Validar se filtros não está vazio e tem valores válidos
    if df.empty:
        print(f"🔍 aplicar_filtros_basicos: DataFrame vazio. Shape: {df.shape}")
        return df
    
    # Filtrar filtros vazios ou None
    filtros_validos = {k: v for k, v in filtros.items() if v is not None and v != '' and v != []}
    
    if not filtros_validos:
        print(f"🔍 aplicar_filtros_basicos: Nenhum filtro válido encontrado. Filtros recebidos: {filtros}")
        return df

    df_filtrado = df.copy()
    filtros_aplicados = []
    
    print(f"🔍 aplicar_filtros_basicos: Aplicando filtros. Shape antes: {df_filtrado.shape}, Filtros válidos: {filtros_validos}")

    # FILTRO 1: Direção (direction, direcao, side)
    direction_filter = (
        filtros_validos.get('direction')
        or filtros_validos.get('directions')
        or filtros_validos.get('side')
        or filtros_validos.get('direcao')
    )

    if direction_filter:
        print(f"   🔍 Filtrando por direção: {direction_filter}")
        direction_col = None
        
        # Procurar coluna de direção em várias variações
        for candidate in ('direction', 'Lado', 'lado', 'Direção', 'direcao', 'Side'):
            if candidate in df_filtrado.columns:
                direction_col = candidate
                break
        
        # Se não encontrou, tentar criar a partir de colunas normalizadas
        if direction_col is None:
            # Verificar se tem coluna normalizada
            if 'direction' not in df_filtrado.columns:
                print(f"   ⚠️ Coluna de direção não encontrada. Colunas disponíveis: {list(df_filtrado.columns)[:10]}")
                # Tentar criar a partir de 'Lado' se existir
                if 'Lado' in df_filtrado.columns:
                    df_filtrado['direction'] = df_filtrado['Lado'].astype(str).str.strip().str.upper().map({
                        'C': 'long', 'COMPRA': 'long', 'COMPRADO': 'long',
                        'V': 'short', 'VENDA': 'short', 'VENDIDO': 'short'
                    }).fillna('long')
                    direction_col = 'direction'
                else:
                    print(f"   ⚠️ Não foi possível aplicar filtro de direção. Continuando sem filtrar por direção.")
            else:
                direction_col = 'direction'

        if direction_col:
            direction_series = df_filtrado[direction_col].astype(str).str.strip()
            mapped_direction = direction_series.str.upper().map({
                'C': 'long', 'COMPRA': 'long', 'COMPRADO': 'long', 'LONG': 'long', 'BUY': 'long',
                'V': 'short', 'VENDA': 'short', 'VENDIDO': 'short', 'SHORT': 'short', 'SELL': 'short'
            })
            
            # Se algum valor não foi mapeado, tentar lowercase
            if mapped_direction.isna().any():
                mapped_direction = mapped_direction.fillna(
                    direction_series.str.lower().map({
                        'c': 'long', 'compra': 'long', 'comprado': 'long', 'long': 'long', 'buy': 'long',
                        'v': 'short', 'venda': 'short', 'vendido': 'short', 'short': 'short', 'sell': 'short'
                    })
                )
            
            df_filtrado['_direction_tmp_'] = mapped_direction.fillna(direction_series.str.lower())
            direction_column_to_use = '_direction_tmp_'

            if isinstance(direction_filter, (list, tuple, set)):
                requested = [str(x).strip().lower() for x in direction_filter if x is not None]
            else:
                requested = [str(direction_filter).strip().lower()] if direction_filter else []

            allowed = set()
            for item in requested:
                if not item:
                    continue
                normalized = item.lower()
                if normalized in _DIRECTION_MAP:
                    mapped = _DIRECTION_MAP[normalized]
                    if not mapped:  # represents 'all'
                        allowed = set()
                        break
                    allowed.update(mapped)
                else:
                    # tentar correspondência parcial
                    if 'compra' in normalized or 'buy' in normalized or 'long' in normalized:
                        allowed.update(_DIRECTION_MAP.get('compra', {'long'}))
                    elif 'venda' in normalized or 'sell' in normalized or 'short' in normalized:
                        allowed.update(_DIRECTION_MAP.get('venda', {'short'}))

            if allowed:
                antes = len(df_filtrado)
                df_filtrado = df_filtrado[df_filtrado[direction_column_to_use].isin(allowed)]
                depois = len(df_filtrado)
                print(f"   ✅ Filtro de direção aplicado: {antes} -> {depois} registros (filtro: {allowed})")
                filtros_aplicados.append(f"direção: {allowed}")

            df_filtrado = df_filtrado.drop(columns=['_direction_tmp_'], errors='ignore')

    # FILTRO 2: Ativo/Símbolo (asset, symbol, ativo, simbolo)
    asset_filter = (
        filtros_validos.get('asset')
        or filtros_validos.get('symbol')
        or filtros_validos.get('ativo')
        or filtros_validos.get('simbolo')
    )
    
    if asset_filter and not df_filtrado.empty:
        print(f"   🔍 Filtrando por ativo: {asset_filter}")
        asset_col = None
        for candidate in ('symbol', 'Ativo', 'ativo', 'asset', 'Asset', 'SYMBOL'):
            if candidate in df_filtrado.columns:
                asset_col = candidate
                break
        
        if asset_col:
            if isinstance(asset_filter, (list, tuple, set)):
                allowed_assets = [str(x).strip() for x in asset_filter if x]
            else:
                allowed_assets = [str(asset_filter).strip()] if asset_filter else []
            
            if allowed_assets:
                antes = len(df_filtrado)
                # Busca case-insensitive
                mask = df_filtrado[asset_col].astype(str).str.strip().str.upper().isin(
                    [a.upper() for a in allowed_assets]
                )
                df_filtrado = df_filtrado[mask]
                depois = len(df_filtrado)
                print(f"   ✅ Filtro de ativo aplicado: {antes} -> {depois} registros (ativos: {allowed_assets})")
                filtros_aplicados.append(f"ativo: {allowed_assets}")
        else:
            print(f"   ⚠️ Coluna de ativo não encontrada. Colunas disponíveis: {list(df_filtrado.columns)[:10]}")

    # FILTRO 3: Estratégia (strategy, estrategia, Estratégia)
    strategy_filter = (
        filtros_validos.get('strategy')
        or filtros_validos.get('estrategia')
        or filtros_validos.get('Estratégia')
    )
    
    if strategy_filter and not df_filtrado.empty:
        print(f"   🔍 Filtrando por estratégia: {strategy_filter}")
        strategy_col = None
        for candidate in ('strategy', 'Estratégia', 'estrategia', 'Strategy', 'STRATEGY'):
            if candidate in df_filtrado.columns:
                strategy_col = candidate
                break
        
        if strategy_col:
            if isinstance(strategy_filter, (list, tuple, set)):
                allowed_strategies = [str(x).strip() for x in strategy_filter if x]
            else:
                allowed_strategies = [str(strategy_filter).strip()] if strategy_filter else []
            
            if allowed_strategies:
                antes = len(df_filtrado)
                # Busca case-insensitive
                mask = df_filtrado[strategy_col].astype(str).str.strip().isin(allowed_strategies)
                df_filtrado = df_filtrado[mask]
                depois = len(df_filtrado)
                print(f"   ✅ Filtro de estratégia aplicado: {antes} -> {depois} registros (estratégias: {allowed_strategies})")
                filtros_aplicados.append(f"estratégia: {allowed_strategies}")
        else:
            print(f"   ⚠️ Coluna de estratégia não encontrada. Colunas disponíveis: {list(df_filtrado.columns)[:10]}")

    # FILTRO 4: Período de datas (date_from, date_to, data_inicio, data_fim)
    if 'entry_date' in df_filtrado.columns and not df_filtrado.empty:
        date_from = filtros_validos.get('date_from') or filtros_validos.get('data_inicio') or filtros_validos.get('date_start')
        date_to = filtros_validos.get('date_to') or filtros_validos.get('data_fim') or filtros_validos.get('date_end')
        
        if date_from or date_to:
            print(f"   🔍 Filtrando por período: {date_from} até {date_to}")
            try:
                df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                antes = len(df_filtrado)
                
                if date_from:
                    date_from_dt = pd.to_datetime(date_from, errors='coerce')
                    if pd.notna(date_from_dt):
                        # CORREÇÃO: Garantir que entry_date está como datetime antes de comparar
                        if not pd.api.types.is_datetime64_any_dtype(df_filtrado['entry_date']):
                            df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                        antes_filtro = len(df_filtrado)
                        df_filtrado = df_filtrado[df_filtrado['entry_date'] >= date_from_dt]
                        depois_filtro = len(df_filtrado)
                        print(f"      📅 Filtro date_from aplicado: {antes_filtro} -> {depois_filtro} registros")
                
                if date_to:
                    date_to_dt = pd.to_datetime(date_to, errors='coerce')
                    if pd.notna(date_to_dt):
                        # CORREÇÃO: Garantir que entry_date está como datetime antes de comparar
                        if not pd.api.types.is_datetime64_any_dtype(df_filtrado['entry_date']):
                            df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                        # Para date_to, incluir o dia inteiro (até 23:59:59)
                        date_to_dt = date_to_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                        antes_filtro = len(df_filtrado)
                        df_filtrado = df_filtrado[df_filtrado['entry_date'] <= date_to_dt]
                        depois_filtro = len(df_filtrado)
                        print(f"      📅 Filtro date_to aplicado: {antes_filtro} -> {depois_filtro} registros")
                
                depois = len(df_filtrado)
                print(f"   ✅ Filtro de data aplicado: {antes} -> {depois} registros")
                filtros_aplicados.append(f"período: {date_from} até {date_to}")
            except Exception as e:
                print(f"   ⚠️ Erro ao aplicar filtro de data: {e}")

    # FILTRO 5: Data específica (specific_date, data_especifica)
    if 'entry_date' in df_filtrado.columns and not df_filtrado.empty:
        specific_date = filtros_validos.get('specific_date') or filtros_validos.get('data_especifica')
        
        if specific_date:
            print(f"   🔍 Filtrando por data específica: {specific_date}")
            try:
                df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                specific_date_dt = pd.to_datetime(specific_date, errors='coerce')
                
                if pd.notna(specific_date_dt):
                    # Filtrar apenas o dia específico (de 00:00:00 até 23:59:59)
                    date_start = specific_date_dt.replace(hour=0, minute=0, second=0, microsecond=0)
                    date_end = specific_date_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
                    
                    antes = len(df_filtrado)
                    mask = (df_filtrado['entry_date'] >= date_start) & (df_filtrado['entry_date'] <= date_end)
                    df_filtrado = df_filtrado[mask]
                    depois = len(df_filtrado)
                    print(f"   ✅ Filtro de data específica aplicado: {antes} -> {depois} registros")
                    filtros_aplicados.append(f"data específica: {specific_date}")
            except Exception as e:
                print(f"   ⚠️ Erro ao aplicar filtro de data específica: {e}")

    # FILTRO 6: Dia da semana (day_of_week, dia_semana)
    if 'entry_date' in df_filtrado.columns and not df_filtrado.empty:
        day_of_week_filter = filtros_validos.get('day_of_week') or filtros_validos.get('dia_semana')
        
        if day_of_week_filter and day_of_week_filter not in ('Todos', 'All', 'todos', 'all', ''):
            print(f"   🔍 Filtrando por dia da semana: {day_of_week_filter}")
            try:
                df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                
                # Mapear nomes de dias (português e inglês)
                day_map = {
                    'Monday': 0, 'Segunda': 0, 'Segunda-feira': 0, 'segunda': 0,
                    'Tuesday': 1, 'Terça': 1, 'Terça-feira': 1, 'terça': 1,
                    'Wednesday': 2, 'Quarta': 2, 'Quarta-feira': 2, 'quarta': 2,
                    'Thursday': 3, 'Quinta': 3, 'Quinta-feira': 3, 'quinta': 3,
                    'Friday': 4, 'Sexta': 4, 'Sexta-feira': 4, 'sexta': 4,
                    'Saturday': 5, 'Sábado': 5, 'sábado': 5,
                    'Sunday': 6, 'Domingo': 6, 'domingo': 6
                }
                
                target_day = day_map.get(str(day_of_week_filter).strip(), None)
                
                if target_day is not None:
                    antes = len(df_filtrado)
                    df_filtrado['_day_of_week'] = df_filtrado['entry_date'].dt.dayofweek
                    df_filtrado = df_filtrado[df_filtrado['_day_of_week'] == target_day]
                    df_filtrado = df_filtrado.drop(columns=['_day_of_week'], errors='ignore')
                    depois = len(df_filtrado)
                    print(f"   ✅ Filtro de dia da semana aplicado: {antes} -> {depois} registros (dia: {day_of_week_filter})")
                    filtros_aplicados.append(f"dia da semana: {day_of_week_filter}")
                else:
                    print(f"   ⚠️ Dia da semana não reconhecido: {day_of_week_filter}")
            except Exception as e:
                print(f"   ⚠️ Erro ao aplicar filtro de dia da semana: {e}")

    # FILTRO 7: Mês (month, mes)
    if 'entry_date' in df_filtrado.columns and not df_filtrado.empty:
        month_filter = filtros_validos.get('month') or filtros_validos.get('mes')
        
        if month_filter and month_filter not in ('Todos', 'All', 'todos', 'all', ''):
            print(f"   🔍 Filtrando por mês: {month_filter}")
            try:
                df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                
                # Mapear nomes de meses (português e inglês) para números
                month_map = {
                    'January': 1, 'Janeiro': 1, 'jan': 1, 'january': 1,
                    'February': 2, 'Fevereiro': 2, 'fev': 2, 'february': 2,
                    'March': 3, 'Março': 3, 'mar': 3, 'march': 3,
                    'April': 4, 'Abril': 4, 'abr': 4, 'april': 4,
                    'May': 5, 'Maio': 5, 'mai': 5, 'may': 5,
                    'June': 6, 'Junho': 6, 'jun': 6, 'june': 6,
                    'July': 7, 'Julho': 7, 'jul': 7, 'july': 7,
                    'August': 8, 'Agosto': 8, 'ago': 8, 'august': 8,
                    'September': 9, 'Setembro': 9, 'set': 9, 'september': 9,
                    'October': 10, 'Outubro': 10, 'out': 10, 'october': 10,
                    'November': 11, 'Novembro': 11, 'nov': 11, 'november': 11,
                    'December': 12, 'Dezembro': 12, 'dez': 12, 'december': 12
                }
                
                # Tentar converter para número (1-12)
                target_month = None
                if isinstance(month_filter, (int, float)):
                    target_month = int(month_filter)
                elif str(month_filter).isdigit():
                    target_month = int(month_filter)
                else:
                    target_month = month_map.get(str(month_filter).strip(), None)
                
                if target_month is not None and 1 <= target_month <= 12:
                    antes = len(df_filtrado)
                    df_filtrado['_month'] = df_filtrado['entry_date'].dt.month
                    df_filtrado = df_filtrado[df_filtrado['_month'] == target_month]
                    df_filtrado = df_filtrado.drop(columns=['_month'], errors='ignore')
                    depois = len(df_filtrado)
                    print(f"   ✅ Filtro de mês aplicado: {antes} -> {depois} registros (mês: {target_month})")
                    filtros_aplicados.append(f"mês: {target_month}")
                else:
                    print(f"   ⚠️ Mês não reconhecido: {month_filter}")
            except Exception as e:
                print(f"   ⚠️ Erro ao aplicar filtro de mês: {e}")

    # FILTRO 8: Faixa de horário (time_from, time_to, hora_inicio, hora_fim)
    if 'entry_date' in df_filtrado.columns and not df_filtrado.empty:
        time_from = filtros_validos.get('time_from') or filtros_validos.get('hora_inicio')
        time_to = filtros_validos.get('time_to') or filtros_validos.get('hora_fim')
        
        if time_from or time_to:
            print(f"   🔍 Filtrando por faixa de horário: {time_from} até {time_to}")
            try:
                df_filtrado['entry_date'] = pd.to_datetime(df_filtrado['entry_date'], errors='coerce')
                
                # Extrair hora e minuto do horário fornecido
                def parse_time(time_str):
                    """Converte string de horário (HH:MM ou HH:MM:SS) para hora e minuto"""
                    if not time_str or time_str in ('--:--', ''):
                        return None
                    try:
                        # Tentar formatos HH:MM ou HH:MM:SS
                        parts = str(time_str).strip().split(':')
                        if len(parts) >= 2:
                            hour = int(parts[0])
                            minute = int(parts[1])
                            if 0 <= hour <= 23 and 0 <= minute <= 59:
                                return (hour, minute)
                    except:
                        pass
                    return None
                
                if time_from:
                    time_from_parsed = parse_time(time_from)
                    if time_from_parsed:
                        hour_from, minute_from = time_from_parsed
                        antes = len(df_filtrado)
                        # Criar máscara para horário >= time_from
                        mask = (df_filtrado['entry_date'].dt.hour > hour_from) | \
                               ((df_filtrado['entry_date'].dt.hour == hour_from) & 
                                (df_filtrado['entry_date'].dt.minute >= minute_from))
                        df_filtrado = df_filtrado[mask]
                        depois = len(df_filtrado)
                        print(f"      🕐 Filtro time_from aplicado: {antes} -> {depois} registros (>= {hour_from:02d}:{minute_from:02d})")
                
                if time_to:
                    time_to_parsed = parse_time(time_to)
                    if time_to_parsed:
                        hour_to, minute_to = time_to_parsed
                        antes = len(df_filtrado)
                        # Criar máscara para horário <= time_to
                        mask = (df_filtrado['entry_date'].dt.hour < hour_to) | \
                               ((df_filtrado['entry_date'].dt.hour == hour_to) & 
                                (df_filtrado['entry_date'].dt.minute <= minute_to))
                        df_filtrado = df_filtrado[mask]
                        depois = len(df_filtrado)
                        print(f"      🕐 Filtro time_to aplicado: {antes} -> {depois} registros (<= {hour_to:02d}:{minute_to:02d})")
                
                if time_from or time_to:
                    filtros_aplicados.append(f"horário: {time_from} até {time_to}")
            except Exception as e:
                print(f"   ⚠️ Erro ao aplicar filtro de horário: {e}")

    print(f"✅ aplicar_filtros_basicos: Filtros aplicados: {filtros_aplicados if filtros_aplicados else 'nenhum'}")
    print(f"   Shape final: {df_filtrado.shape} (antes: {df.shape})")
    
    return df_filtrado

def carregar_csv_trades(file_path_or_file):
    """
    Carrega CSV/Excel da planilha de trades com mapeamento específico e parsing melhorado
    CORREÇÃO: Agora usa a mesma lógica de carregar_csv para suportar todos os formatos (CSV, XLS, XLSX)
    """
    try:
        # CORREÇÃO: Usar a função unificada carregar_csv que suporta todos os formatos
        from FunCalculos import carregar_csv
        df = carregar_csv(file_path_or_file)
        
        # A função carregar_csv já normaliza o DataFrame, então entry_date e pnl já devem existir
        # Mas podemos fazer mapeamentos adicionais se necessário
        
        # Converter direção para formato padrão se ainda não foi feito
        if 'direction' in df.columns:
            # Verificar se já está no formato correto
            sample = df['direction'].dropna().head(5)
            if len(sample) > 0:
                # Se tem valores como 'C' ou 'V', converter
                if any(val in ['C', 'V', 'c', 'v'] for val in sample.astype(str)):
                    df['direction'] = df['direction'].astype(str).str.upper().map({
                        'C': 'long', 'COMPRA': 'long', 'COMPRADO': 'long',
                        'V': 'short', 'VENDA': 'short', 'VENDIDO': 'short'
                    }).fillna(df['direction'])
        
        # Calcular duração em horas se não existir
        if 'entry_date' in df.columns and 'exit_date' in df.columns:
            if df['entry_date'].notna().any() and df['exit_date'].notna().any():
                valid_mask = df['entry_date'].notna() & df['exit_date'].notna()
                if valid_mask.any():
                    df.loc[valid_mask, 'duration_hours'] = (
                        df.loc[valid_mask, 'exit_date'] - df.loc[valid_mask, 'entry_date']
                    ).dt.total_seconds() / 3600
        
        return df
        
    except Exception as e:
        raise ValueError(f"Erro ao processar arquivo de trades: {e}")

# Função carregar_csv_safe melhorada com encoding robusto
def carregar_csv_safe(file_path_or_file):
    """
    CORRIGIDO: Função auxiliar para carregar CSV/Excel com encoding seguro e suporte a múltiplos tipos de arquivo.
    Agora suporta CSV, Excel (.xlsx, .xls, .xlsm) e JSON, além de validar campos obrigatórios.
    CORREÇÃO: Usa a mesma lógica unificada de carregar_csv para garantir padronização.
    """
    try:
        # CORREÇÃO: Usar a função carregar_csv do FunCalculos que foi melhorada e suporta todos os formatos
        from FunCalculos import carregar_csv
        
        # Resetar posição do arquivo se for um objeto file
        if hasattr(file_path_or_file, 'seek'):
            file_path_or_file.seek(0)
        
        # A função carregar_csv já faz toda a normalização e validação
        df = carregar_csv(file_path_or_file)
        
        # CORREÇÃO: Validar campos obrigatórios após carregar (já deve estar normalizado)
        if df.empty:
            raise ValueError("O arquivo está vazio ou não contém dados válidos.")
        
        # Validar que há pelo menos uma coluna de data e uma de PnL (já deve existir após normalização)
        has_date_col = 'entry_date' in df.columns
        has_pnl_col = 'pnl' in df.columns
        
        if not has_date_col:
            raise ValueError("O arquivo não contém coluna de data (entry_date). A normalização deveria ter criado esta coluna.")
        
        if not has_pnl_col:
            raise ValueError("O arquivo não contém coluna de resultado (pnl). A normalização deveria ter criado esta coluna.")
        
        # Validar que há valores válidos
        entry_date_valid = df['entry_date'].notna().sum() if has_date_col else 0
        pnl_valid = df['pnl'].notna().sum() if has_pnl_col else 0
        
        if entry_date_valid == 0:
            raise ValueError("O arquivo não contém datas válidas na coluna 'entry_date'.")
        
        if pnl_valid == 0:
            raise ValueError("O arquivo não contém valores válidos na coluna 'pnl'.")
        
        return df
    except Exception as primary_error:
        print(f"🔍 DEBUG: Fallback para leitura manual do CSV ({primary_error})")
        try:
            encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            formats_to_try = [
                {'skiprows': 0, 'sep': ',', 'encoding': None},
                {'skiprows': 5, 'sep': ';', 'encoding': None, 'decimal': ','},
                {'skiprows': 0, 'sep': ',', 'encoding': None},
                {'skiprows': 5, 'sep': ';', 'encoding': None, 'decimal': ','}
            ]

            df = None
            last_error = primary_error

            for encoding in encodings_to_try:
                for format_config in formats_to_try:
                    try:
                        if hasattr(file_path_or_file, 'read'):
                            file_path_or_file.seek(0)
                            format_config['encoding'] = encoding
                            df = pd.read_csv(file_path_or_file, **format_config)
                        else:
                            format_config['encoding'] = encoding
                            df = pd.read_csv(file_path_or_file, **format_config)

                        expected_columns = ['entry_date', 'exit_date', 'pnl', 'Abertura', 'Fechamento', 'Res. Operação', 'Res. Intervalo']
                        found_columns = [col for col in expected_columns if col in df.columns]
                        if found_columns:
                            break
                    except Exception as e:
                        last_error = e
                        continue
                if df is not None and len(df.columns) > 0:
                    break

            if df is None or len(df.columns) == 0:
                raise ValueError(f"Não foi possível ler o CSV com nenhum encoding/formato. Último erro: {last_error}")
        except Exception as fallback_error:
            print(f"❌ DEBUG: Fallback falhou: {fallback_error}")
            raise ValueError(f"Erro ao processar CSV: {primary_error}. Fallback também falhou: {fallback_error}")

    # Processar datas conforme função original - com verificação de colunas
    if 'Abertura' in df.columns:
        df['Abertura']   = pd.to_datetime(df['Abertura'],   format="%d/%m/%Y %H:%M:%S", errors='coerce')
    if 'Fechamento' in df.columns:
        df['Fechamento'] = pd.to_datetime(df['Fechamento'], format="%d/%m/%Y %H:%M:%S", errors='coerce')

    # Usar função de limpeza para valores numéricos
    numeric_columns = ['Res. Operação', 'Res. Operação (%)', 'Preço Compra', 'Preço Venda',
                      'Preço de Mercado', 'Médio', 'Res. Intervalo', 'Res. Intervalo (%)',
                      'Res. Intervalo Bruto', 'Res. Intervalo Bruto (%)',
                      'Drawdown', 'Ganho Max.', 'Perda Max.', 'Qtd Compra', 'Qtd Venda', 'Total']

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

    df = df.rename(columns=column_mapping)

    if 'direction' in df.columns:
        df['direction'] = df['direction'].map({'C': 'long', 'V': 'short'}).fillna('long')

    if 'pnl' not in df.columns and 'operation_result' in df.columns:
        df['pnl'] = df['operation_result']
    if 'pnl' in df.columns:
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')

    if 'entry_date' in df.columns and 'exit_date' in df.columns:
        try:
            if hasattr(df['entry_date'], 'dtype') and df['entry_date'].dtype == 'object':
                df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
            if hasattr(df['exit_date'], 'dtype') and df['exit_date'].dtype == 'object':
                df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')

            valid_dates = df['entry_date'].notna() & df['exit_date'].notna()
            if valid_dates.any():
                try:
                    duration_series = (df.loc[valid_dates, 'exit_date'] - df.loc[valid_dates, 'entry_date'])
                    df.loc[valid_dates, 'duration_hours'] = duration_series.dt.total_seconds() / 3600
                except Exception as e:
                    print(f"🔍 DEBUG: Erro ao calcular duração: {e}")
        except Exception as e:
            print(f"🔍 DEBUG: Erro ao processar datas: {e}")
            try:
                df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
                df['exit_date'] = pd.to_datetime(df['exit_date'], errors='coerce')
            except Exception:
                pass

    # CORREÇÃO CRÍTICA: Sempre normalizar o DataFrame antes de retornar
    # Isso garante que entry_date e pnl sempre existam no formato correto
    from FunCalculos import _normalize_trades_dataframe
    print(f"🔄 carregar_csv_safe: Normalizando DataFrame antes de retornar...")
    df_original_len = len(df)
    df = _normalize_trades_dataframe(df)
    
    if df.empty:
        print(f"⚠️ carregar_csv_safe: DataFrame ficou vazio após normalização (tinha {df_original_len} linhas)")
    else:
        entry_date_valid = df['entry_date'].notna().sum() if 'entry_date' in df.columns else 0
        pnl_valid = df['pnl'].notna().sum() if 'pnl' in df.columns else 0
        print(f"✅ carregar_csv_safe: DataFrame normalizado - entry_date válidos: {entry_date_valid}/{len(df)}, pnl válidos: {pnl_valid}/{len(df)}")
    
    print(f"🔍 DEBUG: DataFrame final, shape: {df.shape}")
    print(f"🔍 DEBUG: Colunas finais: {df.columns.tolist()}")
    return df

def processar_trades(df: pd.DataFrame, arquivo_para_indices: Dict[int, str] = None) -> List[Dict]:
    """Converte DataFrame em lista de trades para o frontend
    - Inclui também operações em aberto (sem exit_date), usando entry_date como fallback para exit_date
    - Mantém PnL informado no CSV
    """
    trades = []

    print(f"🔍 Processando trades - DataFrame shape: {df.shape}")
    print(f"📅 Colunas disponíveis: {list(df.columns)}")

    # CORREÇÃO CRÍTICA: Normalizar o DataFrame SEMPRE, mesmo se entry_date já existe
    # Quando concatenamos DataFrames, um pode ter 'Abertura' e outro 'entry_date'
    # A normalização garante que todos usem as mesmas colunas padronizadas
    from FunCalculos import _normalize_trades_dataframe
    
    # Verificar se precisa normalizar (se tem colunas não normalizadas como 'Abertura' ou se entry_date está vazio)
    needs_normalization = (
        df.empty or 
        'entry_date' not in df.columns or 
        'pnl' not in df.columns or
        ('Abertura' in df.columns and ('entry_date' not in df.columns or df['entry_date'].isna().all()))
    )
    
    if needs_normalization:
        print(f"🔄 Normalizando DataFrame em processar_trades...")
        df = _normalize_trades_dataframe(df)
        if df.empty:
            print("⚠️ DataFrame vazio após normalização")
            return trades
        print(f"✅ DataFrame normalizado. entry_date válidos: {df['entry_date'].notna().sum() if 'entry_date' in df.columns else 0}/{len(df)}")

    # Verificar se a coluna mínima necessária existe
    required_columns = ['entry_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Colunas faltando: {missing_columns}. Colunas disponíveis: {list(df.columns)}")
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
        filename = None
        if arquivo_para_indices and idx in arquivo_para_indices:
            filename = arquivo_para_indices[idx]
        elif 'source_file' in df.columns:
            filename = row.get('source_file')

        if filename:
            filename_str = str(filename)
            strategy = Path(filename_str).stem

        qty_buy_raw = row.get('qty_buy', 0)
        qty_sell_raw = row.get('qty_sell', 0)

        qty_buy = int(qty_buy_raw) if pd.notna(qty_buy_raw) else 0
        qty_sell = int(qty_sell_raw) if pd.notna(qty_sell_raw) else 0
        # Somar buy/sell; se ambos 0, tentar usar outras colunas
        quantity_total = qty_buy + qty_sell
        if quantity_total == 0:
            for fallback in ('quantity', 'contracts', 'position', 'Position', 'Qtd', 'Qtd Total'):
                if fallback in row.index and pd.notna(row[fallback]):
                    try:
                        quantity_total = int(float(row[fallback]))
                        break
                    except (ValueError, TypeError):
                        continue
        if quantity_total == 0 and 'qty' in row.index and pd.notna(row['qty']):
            try:
                quantity_total = int(float(row['qty']))
            except (ValueError, TypeError):
                quantity_total = 0

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
            "source_file": filename,
            "quantity_total": quantity_total,
            "quantity_compra": qty_buy,
            "quantity_venda": qty_sell,
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
    
    # CORREÇÃO: Detectar coluna de PnL automaticamente
    from FunCalculos import _detect_pnl_column
    pnl_col = _detect_pnl_column(df)
    if pnl_col is None:
        return {}
    
    df_valid = df.dropna(subset=['entry_date', pnl_col])
    
    if df_valid.empty:
        return {}
    
    # Por dia da semana
    df_valid['day_of_week'] = df_valid['entry_date'].dt.day_name()
    day_stats = df_valid.groupby('day_of_week')[pnl_col].agg(['count', 'sum', 'mean']).round(2)
    
    # Por mês - converter Period para string
    df_valid['month'] = df_valid['entry_date'].dt.to_period('M').astype(str)
    month_stats = df_valid.groupby('month')[pnl_col].agg(['count', 'sum', 'mean']).round(2)
    
    # Por hora
    df_valid['hour'] = df_valid['entry_date'].dt.hour
    hour_stats = df_valid.groupby('hour')[pnl_col].agg(['count', 'sum', 'mean']).round(2)
    
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
        # Tratar valores especiais
        if np.isnan(obj) or np.isinf(obj):
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
        if isinstance(item_value, float) and (np.isnan(item_value) or np.isinf(item_value)):
            return None
        return item_value
    elif isinstance(obj, float):
        # Tratar valores especiais para floats Python também
        if np.isnan(obj) or np.isinf(obj):
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

def _extrair_taxas_do_request(req) -> tuple:
    """
    Extrai taxas de corretagem e emolumentos do request
    CORREÇÃO: Suporta formato separado, formato antigo e configurações complexas
    Suporta:
    - Taxas simples (float)
    - Configurações com método (fixed/percentage) e valor
    - Configurações por ativo
    Retorna (taxa_corretagem, taxa_emolumentos) onde cada uma pode ser None ou um dict com método e valor
    """
    taxa_corretagem = None
    taxa_emolumentos = None
    
    # Tentar extrair do form
    taxa_corretagem_str = (req.form.get('taxa_corretagem') or 
                          req.form.get('corretagem') or 
                          req.form.get('brokerage') or
                          req.form.get('backtest_commission'))
    taxa_emolumentos_str = (req.form.get('taxa_emolumentos') or 
                           req.form.get('emolumentos') or 
                           req.form.get('emoluments') or
                           req.form.get('backtest_fees'))
    
    # Tentar extrair do JSON
    json_data = None
    if req.is_json:
        json_data = req.get_json(silent=True)
        if json_data:
            # Extrair corretagem do JSON
            if not taxa_corretagem_str:
                # Tentar diferentes formatos
                if 'corretagem' in json_data:
                    corretagem_data = json_data['corretagem']
                    if isinstance(corretagem_data, dict):
                        # Formato: {"method": "fixed", "value": 0.5}
                        taxa_corretagem_str = str(corretagem_data.get('value', 0))
                    else:
                        taxa_corretagem_str = str(corretagem_data)
                elif 'brokerage' in json_data:
                    brokerage_data = json_data['brokerage']
                    if isinstance(brokerage_data, dict):
                        taxa_corretagem_str = str(brokerage_data.get('value', 0))
                    else:
                        taxa_corretagem_str = str(brokerage_data)
                else:
                    taxa_corretagem_str = (json_data.get('taxa_corretagem') or 
                                          json_data.get('backtest_commission'))
            
            # Extrair emolumentos do JSON
            if not taxa_emolumentos_str:
                if 'emolumentos' in json_data:
                    emolumentos_data = json_data['emolumentos']
                    if isinstance(emolumentos_data, dict):
                        taxa_emolumentos_str = str(emolumentos_data.get('value', 0))
                    else:
                        taxa_emolumentos_str = str(emolumentos_data)
                elif 'emoluments' in json_data:
                    emoluments_data = json_data['emoluments']
                    if isinstance(emoluments_data, dict):
                        taxa_emolumentos_str = str(emoluments_data.get('value', 0))
                    else:
                        taxa_emolumentos_str = str(emoluments_data)
                else:
                    taxa_emolumentos_str = (json_data.get('taxa_emolumentos') or 
                                            json_data.get('backtest_fees'))
    
    # Converter para float se fornecido
    if taxa_corretagem_str:
        try:
            taxa_corretagem = float(taxa_corretagem_str)
            print(f"💼 Taxa de corretagem extraída do request: R$ {taxa_corretagem:.2f}")
        except (ValueError, TypeError):
            print(f"⚠️ Taxa de corretagem inválida: {taxa_corretagem_str}")
    
    if taxa_emolumentos_str:
        try:
            taxa_emolumentos = float(taxa_emolumentos_str)
            print(f"💼 Taxa de emolumentos extraída do request: R$ {taxa_emolumentos:.2f}")
        except (ValueError, TypeError):
            print(f"⚠️ Taxa de emolumentos inválida: {taxa_emolumentos_str}")
    
    return taxa_corretagem, taxa_emolumentos

def calcular_custos_operacionais(df: pd.DataFrame, taxa_corretagem: float = None, taxa_emolumentos: float = None) -> Dict[str, Any]:
    """
    Calcula custos operacionais estimados
    CORREÇÃO: Separa corretamente corretagem e emolumentos
    
    Args:
        df: DataFrame com trades
        taxa_corretagem: Taxa de corretagem (por roda ou por trade, dependendo do valor)
        taxa_emolumentos: Taxa de emolumentos (percentual ou fixa por roda, dependendo do valor)
    """
    if df.empty:
        return {}
    
    # CORREÇÃO: Usar valores padrão se não fornecidos
    if taxa_corretagem is None:
        taxa_corretagem = 0.50  # R$ 0,50 por roda (padrão mercado brasileiro)
    if taxa_emolumentos is None:
        taxa_emolumentos = 0.03  # R$ 0,03 por roda (padrão mercado brasileiro)
    
    # CORREÇÃO: Verificar se as colunas existem antes de usar
    required_cols = []
    if 'entry_price' in df.columns and 'exit_price' in df.columns:
        required_cols = ['entry_price', 'exit_price']
    elif 'Preço Compra' in df.columns and 'Preço Venda' in df.columns:
        # Usar colunas originais se não tiver as normalizadas
        required_cols = ['Preço Compra', 'Preço Venda']
    else:
        # Se não tem colunas de preço, usar todas as linhas (assumir que são válidas)
        df_valid = df.copy()
        total_trades = len(df_valid)
        # Retornar valores básicos sem cálculos de valor operado
        quantidade_rodas = total_trades * 2  # Assumir 2 rodas por trade
        custo_corretagem = quantidade_rodas * taxa_corretagem if taxa_corretagem < 1.0 else total_trades * taxa_corretagem
        custo_emolumentos = quantidade_rodas * taxa_emolumentos
        custo_total = custo_corretagem + custo_emolumentos
        
        return {
            "total_trades": total_trades,
            "quantidade_rodas": int(quantidade_rodas),
            "valor_total_operado": 0.0,  # Não foi possível calcular
            "custo_corretagem": round(custo_corretagem, 2),
            "custo_emolumentos": round(custo_emolumentos, 2),
            "custo_total": round(custo_total, 2),
            "custo_por_trade": round(custo_total / total_trades, 2) if total_trades > 0 else 0.0
        }
    
    df_valid = df.dropna(subset=required_cols)
    total_trades = len(df_valid)
    
    if total_trades == 0:
        return {
            "total_trades": 0,
            "valor_total_operado": 0.0,
            "custo_corretagem": 0.0,
            "custo_emolumentos": 0.0,
            "custo_total": 0.0,
            "custo_por_trade": 0.0
        }
    
    # Calcular quantidade de rodas
    quantidade_rodas = 0
    if 'quantity' in df_valid.columns:
        quantidade_rodas = df_valid['quantity'].sum() * 2  # Entrada + saída
    elif 'qty_buy' in df_valid.columns and 'qty_sell' in df_valid.columns:
        quantidade_rodas = (df_valid['qty_buy'].sum() + df_valid['qty_sell'].sum())
    elif 'Qtd Compra' in df_valid.columns and 'Qtd Venda' in df_valid.columns:
        quantidade_rodas = (df_valid['Qtd Compra'].sum() + df_valid['Qtd Venda'].sum())
    else:
        # Fallback: assumir 2 rodas por trade (entrada + saída)
        quantidade_rodas = total_trades * 2
    
    # Calcular valor total operado
    # CORREÇÃO: Verificar se as colunas de preço existem antes de usar
    has_entry_price = 'entry_price' in df_valid.columns
    has_exit_price = 'exit_price' in df_valid.columns
    has_preco_compra = 'Preço Compra' in df_valid.columns
    has_preco_venda = 'Preço Venda' in df_valid.columns
    
    if has_entry_price and has_exit_price:
        if 'position_size' in df_valid.columns:
            valor_entrada = df_valid['entry_price'] * df_valid['position_size']
            valor_saida = df_valid['exit_price'] * df_valid['position_size']
        elif 'quantity' in df_valid.columns:
            valor_entrada = df_valid['entry_price'] * df_valid['quantity']
            valor_saida = df_valid['exit_price'] * df_valid['quantity']
        else:
            # Fallback: assumir 1 contrato
            valor_entrada = df_valid['entry_price']
            valor_saida = df_valid['exit_price']
        valor_total_operado = float((valor_entrada + valor_saida).sum())
    elif has_preco_compra and has_preco_venda:
        # Usar colunas originais se não tiver as normalizadas
        if 'Qtd Compra' in df_valid.columns:
            valor_entrada = df_valid['Preço Compra'] * df_valid['Qtd Compra']
            valor_saida = df_valid['Preço Venda'] * df_valid['Qtd Venda'] if 'Qtd Venda' in df_valid.columns else df_valid['Preço Venda'] * df_valid['Qtd Compra']
        else:
            valor_entrada = df_valid['Preço Compra']
            valor_saida = df_valid['Preço Venda']
        valor_total_operado = float((valor_entrada + valor_saida).sum())
    else:
        # Se não tem colunas de preço, não é possível calcular valor operado
        valor_total_operado = 0.0
    
    # CORREÇÃO: Calcular corretagem (sempre por roda)
    # Se taxa_corretagem < 1, é por roda. Se >= 1, pode ser por trade
    if taxa_corretagem < 1.0:
        # Taxa por roda
        custo_corretagem = quantidade_rodas * taxa_corretagem
    else:
        # Taxa por trade (assumir que é o valor total para entrada + saída)
        custo_corretagem = total_trades * taxa_corretagem
    
    # CORREÇÃO: Calcular emolumentos (pode ser percentual ou fixo por roda)
    if taxa_emolumentos < 1.0:
        # Se < 1, pode ser percentual (ex: 0.03 = 3%) ou fixo por roda (ex: 0.03 = R$ 0,03)
        # Tentar calcular como percentual primeiro
        if valor_total_operado > 0:
            # Assumir que é percentual se o valor operado for grande
            if valor_total_operado > 10000:
                custo_emolumentos = valor_total_operado * (taxa_emolumentos / 100.0)
            else:
                # Se valor operado é pequeno, provavelmente é fixo por roda
                custo_emolumentos = quantidade_rodas * taxa_emolumentos
        else:
            # Se não tem valor operado, usar por roda
            custo_emolumentos = quantidade_rodas * taxa_emolumentos
    else:
        # Taxa fixa por roda (valores >= 1)
        custo_emolumentos = quantidade_rodas * taxa_emolumentos
    
    custo_total = custo_corretagem + custo_emolumentos
    
    print(f"💼 calcular_custos_operacionais: Corretagem: R$ {custo_corretagem:.2f} ({quantidade_rodas:.0f} rodas × R$ {taxa_corretagem:.2f}), Emolumentos: R$ {custo_emolumentos:.2f}")
    
    return {
        "total_trades": total_trades,
        "quantidade_rodas": int(quantidade_rodas),
        "valor_total_operado": round(valor_total_operado, 2),
        "custo_corretagem": round(custo_corretagem, 2),
        "custo_emolumentos": round(custo_emolumentos, 2),
        "custo_total": round(custo_total, 2),
        "custo_por_trade": round(custo_total / total_trades, 2) if total_trades > 0 else 0.0
    }

# ============ FUNÇÕES PARA MÉTRICAS DIÁRIAS ============

def calcular_metricas_diarias(df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula métricas diárias baseadas nas trades"""
    if df.empty:
        return {}
    
    # CORREÇÃO: Normalizar o DataFrame se necessário
    from FunCalculos import _normalize_trades_dataframe
    if 'entry_date' not in df.columns or 'pnl' not in df.columns:
        df = _normalize_trades_dataframe(df)
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
    
    # CORREÇÃO: Normalizar o DataFrame se necessário
    from FunCalculos import _normalize_trades_dataframe
    if 'entry_date' not in df.columns or 'pnl' not in df.columns:
        print("  🔄 Normalizando DataFrame...")
        df = _normalize_trades_dataframe(df)
        if df.empty:
            print("⚠️ DataFrame vazio após normalização")
            return pd.DataFrame()
        print(f"  ✅ DataFrame normalizado. Colunas: {df.columns.tolist()}")
    
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
    
    # CORREÇÃO: Normalizar o DataFrame se necessário
    from FunCalculos import _normalize_trades_dataframe
    if 'entry_date' not in df.columns or 'pnl' not in df.columns:
        df = _normalize_trades_dataframe(df)
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
    
    # CORREÇÃO: Payoff Ratio (Ganho médio / Perda média) com validação
    wins_pnl = df_valid[df_valid['pnl'] > 0]['pnl'].dropna()
    losses_pnl = df_valid[df_valid['pnl'] < 0]['pnl'].dropna()
    
    avg_win = wins_pnl.mean() if len(wins_pnl) > 0 else 0.0
    avg_loss = abs(losses_pnl.mean()) if len(losses_pnl) > 0 else 0.0
    
    # Garantir valores válidos
    if pd.isna(avg_win) or np.isinf(avg_win):
        avg_win = 0.0
    if pd.isna(avg_loss) or np.isinf(avg_loss):
        avg_loss = 0.0
    
    # Calcular payoff corretamente
    if avg_loss > 0 and not pd.isna(avg_loss) and not np.isinf(avg_loss):
        payoff_ratio = avg_win / avg_loss if not pd.isna(avg_win) and not np.isinf(avg_win) else 0.0
    else:
        payoff_ratio = 0.0
    
    # Garantir que payoff seja válido
    if pd.isna(payoff_ratio) or np.isinf(payoff_ratio):
        payoff_ratio = 0.0
    
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
    
    # SHARPE RATIO CORRIGIDO - Usar desvio padrão corretamente
    # Calcular retornos dos trades individuais (como no FunCalculos.py)
    returns = df_valid['pnl'].values
    mean_return = np.mean(returns) if len(returns) > 0 else 0
    # CORREÇÃO: Usar desvio padrão amostral (ddof=1) para correção de Bessel
    # Isso é importante para amostras pequenas (correção de viés)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0
    
    # CORREÇÃO: Calcular métricas estatísticas adicionais usando desvio padrão
    volatility = std_return
    variance = np.var(returns, ddof=1) if len(returns) > 1 else 0
    coefficient_of_variation = (volatility / abs(mean_return) * 100) if mean_return != 0 else 0
    
    # CORREÇÃO: Ajustar CDI para o período dos retornos
    # CDI é anual (12%), mas retornos são por trade
    # Ajustamos proporcionalmente ao número de trades no período
    cdi_annual = 0.12  # Taxa anual (12% ao ano)
    # Para retornos por trade, ajustamos o CDI baseado no número de trades
    # Assumindo ~252 dias úteis por ano e média de trades por dia
    if days_traded > 0:
        trades_per_day = total_trades / days_traded
        # Ajustar CDI para retorno por trade: CDI_por_trade = CDI_anual / (252 * trades_por_dia)
        # Simplificado: usar CDI diretamente se não temos informação suficiente
        cdi = cdi_annual / 252 if trades_per_day > 0 else cdi_annual
    else:
        cdi = cdi_annual
    
    sharpe_ratio = ((mean_return - cdi) / std_return) if std_return > 0 else 0
    
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
            "volatilidade": round(volatility, 2),
            "variancia": round(variance, 2),
            "coeficiente_variacao": round(coefficient_of_variation, 2),
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
            "payoff_diario": round(daily_avg_win / daily_avg_loss if daily_avg_loss > 0 and not pd.isna(daily_avg_loss) and not np.isinf(daily_avg_loss) and not pd.isna(daily_avg_win) and not np.isinf(daily_avg_win) else 0.0, 2),
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
    
    column_candidates = [
        'operation_result', 'pnl', 'resultado',
        'Res. Operação', 'Res. Operacao', 'Res. Operação Bruta',
        'Res. Intervalo', 'Res. Intervalo Bruto', 'Total'
    ]

    for col_name in column_candidates:
        if col_name in df.columns:
            if col_name not in ['operation_result', 'pnl', 'resultado']:
                tmp_col = f"_resultado_tmp_{col_name}"
                df[tmp_col] = pd.to_numeric(df[col_name], errors='coerce')
                resultado_col = tmp_col
            else:
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
    resultado_final = {
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
            "pior_operacao": round(pior_operacao, 2) if total_operacoes > 0 else 0.0,
            "melhor_operacao": round(melhor_operacao, 2) if total_operacoes > 0 else 0.0,
            "pior_dia": round(pior_dia, 2) if total_dias > 0 else 0.0,
            "melhor_dia": round(melhor_dia, 2) if total_dias > 0 else 0.0,
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

    for colname in list(df.columns):
        if str(colname).startswith("_resultado_tmp_") or str(colname).startswith("_data_tmp_"):
            df.drop(columns=[colname], inplace=True, errors='ignore')

    return resultado_final

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
        
        # CORREÇÃO: Normalizar o DataFrame consolidado antes de calcular disciplina
        from FunCalculos import _normalize_trades_dataframe
        if 'entry_date' not in df_consolidado.columns or 'pnl' not in df_consolidado.columns:
            print(f"🔄 api_disciplina_completa: Normalizando DataFrame consolidado (shape: {df_consolidado.shape})...")
            df_consolidado = _normalize_trades_dataframe(df_consolidado)
            if df_consolidado.empty:
                return jsonify({"error": "Após normalização, o arquivo ficou vazio. Verifique os dados."}), 400
        
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
        
        # CORREÇÃO CRÍTICA: Normalizar CADA DataFrame individualmente logo após carregar
        # Isso garante que todos tenham entry_date e pnl ANTES de serem processados individualmente
        from FunCalculos import _normalize_trades_dataframe
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_safe(arquivo)
                # Normalizar imediatamente após carregar
                print(f"🔄 api_tabela_multipla: Normalizando arquivo único '{arquivo.filename}' após carregar...")
                if 'entry_date' not in df.columns or 'pnl' not in df.columns or (hasattr(df, 'entry_date') and df['entry_date'].isna().all() if 'entry_date' in df.columns else False):
                    df = _normalize_trades_dataframe(df)
                    entry_date_valid = df['entry_date'].notna().sum() if 'entry_date' in df.columns else 0
                    print(f"   ✅ Arquivo único normalizado: entry_date válidos: {entry_date_valid}/{len(df)}")
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_safe(arquivo)
                    # CORREÇÃO CRÍTICA: Normalizar imediatamente após carregar CADA arquivo
                    print(f"🔄 api_tabela_multipla: Normalizando '{arquivo.filename}' após carregar...")
                    needs_norm = 'entry_date' not in df.columns or 'pnl' not in df.columns
                    if not needs_norm and 'entry_date' in df.columns:
                        needs_norm = df['entry_date'].isna().all()
                    if needs_norm:
                        df_before = df.copy()
                        df = _normalize_trades_dataframe(df)
                        if df.empty:
                            print(f"   ⚠️ Arquivo '{arquivo.filename}' ficou vazio após normalização (tinha {len(df_before)} linhas)")
                        else:
                            entry_date_valid = df['entry_date'].notna().sum() if 'entry_date' in df.columns else 0
                            pnl_valid = df['pnl'].notna().sum() if 'pnl' in df.columns else 0
                            print(f"   ✅ Arquivo normalizado: entry_date válidos: {entry_date_valid}/{len(df)}, pnl válidos: {pnl_valid}/{len(df)}")
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_safe(path)
            # Normalizar imediatamente após carregar
            print(f"🔄 api_tabela_multipla: Normalizando arquivo por path '{path}' após carregar...")
            if 'entry_date' not in df.columns or 'pnl' not in df.columns or (hasattr(df, 'entry_date') and df['entry_date'].isna().all() if 'entry_date' in df.columns else False):
                df = _normalize_trades_dataframe(df)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo enviado. Use 'file' para um arquivo ou 'files' para múltiplos"}), 400
        
        # CORREÇÃO: Extrair filtros ANTES de processar
        filtros = _parse_filters_from_request(request)
        print(f"🔍 Filtros recebidos: {filtros}")
        
        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # CORREÇÃO: Extrair taxas usando função auxiliar
        taxa_corretagem, taxa_emolumentos = _extrair_taxas_do_request(request)
        
        # Processar cada arquivo individualmente
        resultados_individuais = {}
        print(f"🔍 Processando {len(dataframes)} arquivos individualmente:")
        for i, (df, nome_arquivo) in enumerate(zip(dataframes, arquivos_processados)):
            try:
                print(f"  📁 Arquivo {i+1}/{len(dataframes)}: {nome_arquivo}")
                print(f"     📊 Registros: {len(df)}")
                print(f"     📅 Colunas: {list(df.columns)}")
                
                # CORREÇÃO CRÍTICA: Normalizar o DataFrame SEMPRE, não apenas se faltar
                # Isso garante que sempre tenhamos entry_date, pnl, etc. no formato correto
                from FunCalculos import _normalize_trades_dataframe
                try:
                    print(f"     🔄 Normalizando DataFrame (shape antes: {df.shape})...")
                    print(f"        Colunas antes: {list(df.columns)[:5]}...")  # Primeiras 5 colunas
                    
                    # SEMPRE normalizar, mesmo se as colunas já existem (para garantir formato correto)
                    df_original_len = len(df)
                    df = _normalize_trades_dataframe(df)
                    
                    if df.empty:
                        print(f"     ⚠️ DataFrame vazio após normalização (tinha {df_original_len} linhas antes)")
                        resultados_individuais[nome_arquivo] = {
                            "error": f"Após normalização, o arquivo ficou vazio. Verifique se há valores válidos nas colunas 'Abertura' e de resultado (Res. Intervalo, Res. Operação, etc.).",
                            "info_arquivo": {
                                "nome_arquivo": nome_arquivo,
                                "total_registros": df_original_len
                            }
                        }
                        continue
                    
                    # Verificar se entry_date foi criado e tem valores válidos
                    has_entry_date = 'entry_date' in df.columns
                    has_pnl = 'pnl' in df.columns
                    entry_date_valid = df['entry_date'].notna().sum() if has_entry_date else 0
                    pnl_valid = df['pnl'].notna().sum() if has_pnl else 0
                    
                    print(f"     ✅ DataFrame normalizado (shape depois: {df.shape})")
                    print(f"        entry_date existe: {has_entry_date}, válidos: {entry_date_valid}/{len(df)}")
                    print(f"        pnl existe: {has_pnl}, válidos: {pnl_valid}/{len(df)}")
                    print(f"        Colunas depois: {list(df.columns)}")
                    
                    # Se não tem entry_date válido, tentar diagnosticar
                    if not has_entry_date or entry_date_valid == 0:
                        print(f"     ⚠️ AVISO: entry_date não criado ou sem valores válidos!")
                        if 'Abertura' in df.columns:
                            print(f"        Coluna 'Abertura' existe. Primeiros valores:")
                            print(f"        {df['Abertura'].head(3).tolist()}")
                        else:
                            print(f"        Coluna 'Abertura' NÃO existe. Colunas disponíveis: {list(df.columns)}")
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"     ❌ Erro ao normalizar DataFrame: {e}")
                    print(f"     Detalhes: {error_details}")
                    resultados_individuais[nome_arquivo] = {
                        "error": f"Erro ao normalizar dados: {str(e)}. Verifique se o arquivo está no formato correto.",
                        "info_arquivo": {
                            "nome_arquivo": nome_arquivo,
                            "total_registros": len(df) if 'df' in locals() else 0,
                            "colunas_originais": list(df.columns) if 'df' in locals() and not df.empty else []
                        }
                    }
                    continue

                # DEBUG: Verificar padronização do drawdown (após normalização)
                try:
                    debug_drawdown_calculation(df)
                except Exception as e:
                    print(f"     ⚠️ Erro no debug_drawdown_calculation: {e}")
                    # Continuar mesmo se o debug falhar
                
                # Após normalização, entry_date e pnl já devem existir
                # Validar que temos as colunas necessárias
                has_entry_date_col = 'entry_date' in df.columns
                has_pnl_col = 'pnl' in df.columns
                entry_date_valid_count = df['entry_date'].notna().sum() if has_entry_date_col else 0
                pnl_valid_count = df['pnl'].notna().sum() if has_pnl_col else 0
                
                # CORREÇÃO: Se entry_date existe mas está vazio, tentar recriar a partir de Abertura
                if has_entry_date_col and entry_date_valid_count == 0:
                    print(f"     ⚠️ entry_date existe mas está vazio. Tentando recriar a partir de 'Abertura'...")
                    if 'Abertura' in df.columns:
                        try:
                            # Tentar múltiplos formatos
                            for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                                df['entry_date'] = pd.to_datetime(df['Abertura'], format=fmt, errors='coerce')
                                if df['entry_date'].notna().any():
                                    entry_date_valid_count = df['entry_date'].notna().sum()
                                    print(f"     ✅ Recriado entry_date usando formato '{fmt}' ({entry_date_valid_count} valores válidos)")
                                    break
                            # Se ainda não funcionou, tentar detecção automática
                            if entry_date_valid_count == 0:
                                df['entry_date'] = pd.to_datetime(df['Abertura'], errors='coerce', infer_datetime_format=True)
                                entry_date_valid_count = df['entry_date'].notna().sum()
                                if entry_date_valid_count > 0:
                                    print(f"     ✅ Recriado entry_date via detecção automática ({entry_date_valid_count} valores válidos)")
                        except Exception as e:
                            print(f"     ⚠️ Erro ao recriar entry_date: {e}")
                
                # Validar colunas obrigatórias
                if not has_entry_date_col:
                    print(f"     ❌ Coluna 'entry_date' não existe após normalização!")
                    print(f"        Colunas disponíveis: {list(df.columns)}")
                    resultados_individuais[nome_arquivo] = {
                        "error": "O arquivo não contém a coluna 'entry_date' obrigatória. O backend tenta mapear automaticamente a coluna 'Abertura' para 'entry_date', mas isso pode ter falhado. Verifique se: A coluna 'Abertura' contém datas válidas; O formato da data está correto; Não há valores nulos na coluna de data.",
                        "info_arquivo": {
                            "nome_arquivo": nome_arquivo,
                            "total_registros": len(df),
                            "colunas_disponiveis": list(df.columns),
                            "tem_abertura": 'Abertura' in df.columns
                        }
                    }
                    continue
                
                if not has_pnl_col:
                    print(f"     ❌ Coluna 'pnl' não existe após normalização!")
                    resultados_individuais[nome_arquivo] = {
                        "error": "Coluna 'pnl' não encontrada após normalização.",
                        "info_arquivo": {
                            "nome_arquivo": nome_arquivo,
                            "total_registros": len(df),
                            "colunas_disponiveis": list(df.columns)
                        }
                    }
                    continue
                
                # CORREÇÃO CRÍTICA: Se entry_date existe mas está vazio, tentar UMA ÚLTIMA VEZ recriar
                # Isso é importante porque a normalização pode ter falhado silenciosamente
                if entry_date_valid_count == 0:
                    print(f"     ⚠️ AVISO: entry_date existe mas está vazio (todos NaT). Tentando recriar UMA ÚLTIMA VEZ...")
                    
                    # Tentar recriar usando a coluna Abertura original (se ainda existir)
                    if 'Abertura' in df.columns:
                        print(f"     🔄 Tentativa final: recriando entry_date a partir de 'Abertura'...")
                        try:
                            # Verificar se Abertura já é datetime
                            if pd.api.types.is_datetime64_any_dtype(df['Abertura']):
                                df['entry_date'] = df['Abertura']
                                entry_date_valid_count = df['entry_date'].notna().sum()
                                if entry_date_valid_count > 0:
                                    print(f"     ✅ SUCESSO! entry_date recriado diretamente de 'Abertura' ({entry_date_valid_count} valores válidos)")
                                else:
                                    print(f"     ❌ Abertura é datetime mas está vazia")
                            else:
                                # Tentar todos os formatos novamente
                                for fmt in ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"]:
                                    df['entry_date'] = pd.to_datetime(df['Abertura'], format=fmt, errors='coerce')
                                    entry_date_valid_count = df['entry_date'].notna().sum()
                                    if entry_date_valid_count > 0:
                                        print(f"     ✅ SUCESSO! entry_date recriado usando formato '{fmt}' ({entry_date_valid_count} valores válidos)")
                                        break
                                
                                # Se ainda não funcionou, tentar detecção automática
                                if entry_date_valid_count == 0:
                                    df['entry_date'] = pd.to_datetime(df['Abertura'], errors='coerce')
                                    entry_date_valid_count = df['entry_date'].notna().sum()
                                    if entry_date_valid_count > 0:
                                        print(f"     ✅ SUCESSO! entry_date recriado via detecção automática ({entry_date_valid_count} valores válidos)")
                        except Exception as e:
                            print(f"     ❌ Erro na tentativa final: {e}")
                    
                    # Se ainda não tem valores válidos após todas as tentativas, bloquear
                    if entry_date_valid_count == 0:
                        print(f"     ❌ Todas as tentativas falharam. entry_date continua vazio.")
                        resultados_individuais[nome_arquivo] = {
                            "error": "O arquivo não contém a coluna 'entry_date' obrigatória. O backend tenta mapear automaticamente a coluna 'Abertura' para 'entry_date', mas isso pode ter falhado. Verifique se: A coluna 'Abertura' contém datas válidas; O formato da data está correto; Não há valores nulos na coluna de data.",
                            "info_arquivo": {
                                "nome_arquivo": nome_arquivo,
                                "total_registros": len(df),
                                "colunas_disponiveis": list(df.columns),
                                "tem_abertura": 'Abertura' in df.columns,
                                "entry_date_vazio": True,
                                "abertura_sample": df['Abertura'].head(3).tolist() if 'Abertura' in df.columns else None
                            }
                        }
                        continue
                    else:
                        print(f"     ✅ entry_date recriado com sucesso! Continuando processamento...")

                # CORREÇÃO CRÍTICA: Aplicar filtros ANTES de processar
                # Os filtros devem ser aplicados após normalização mas antes de processar
                if filtros:
                    print(f"     🔍 Aplicando filtros ao arquivo {nome_arquivo}...")
                    df_antes_filtro = len(df)
                    df = aplicar_filtros_basicos(df, filtros)
                    df = df.reset_index(drop=True)
                    df_depois_filtro = len(df)
                    print(f"     ✅ Filtros aplicados: {df_antes_filtro} -> {df_depois_filtro} registros")
                    
                    # Se após filtros o DataFrame ficou vazio, pular este arquivo
                    if df.empty:
                        print(f"     ⚠️ DataFrame ficou vazio após aplicar filtros. Pulando arquivo {nome_arquivo}.")
                        resultados_individuais[nome_arquivo] = {
                            "error": "Nenhum registro corresponde aos filtros aplicados.",
                            "info_arquivo": {
                                "nome_arquivo": nome_arquivo,
                                "total_registros_antes_filtro": df_antes_filtro
                            }
                        }
                        continue

                # CORREÇÃO: Passar taxas customizadas para processar_backtest_completo
                # Se taxas foram fornecidas, passá-las. Caso contrário, None (cálculo automático)
                resultado_individual = processar_backtest_completo(
                    df, 
                    capital_inicial=capital_inicial, 
                    cdi=cdi,
                    taxa_corretagem=taxa_corretagem,
                    taxa_emolumentos=taxa_emolumentos
                )

                # Garantir compatibilidade de chaves no resultado individual (para o frontend)
                try:
                    # Copiar em camelCase as seções principais
                    if 'Day of Week Analysis' in resultado_individual:
                        resultado_individual['day_of_week'] = resultado_individual['Day of Week Analysis']
                    if 'Monthly Analysis' in resultado_individual:
                        resultado_individual['monthly'] = resultado_individual['Monthly Analysis']
                    if 'Equity Curve Data' in resultado_individual:
                        resultado_individual['equity_curve_data'] = resultado_individual['Equity Curve Data']
                    if 'Position Sizing' in resultado_individual:
                        resultado_individual['position_sizing'] = resultado_individual['Position Sizing']
                        resultado_individual['positionSizing'] = resultado_individual['Position Sizing']
                    if 'Trade Duration' in resultado_individual:
                        resultado_individual['trade_duration'] = resultado_individual['Trade Duration']
                        resultado_individual['tradeDuration'] = resultado_individual['Trade Duration']
                    if 'Operational Costs' in resultado_individual:
                        resultado_individual['operational_costs'] = resultado_individual['Operational Costs']
                        resultado_individual['operationalCosts'] = resultado_individual['Operational Costs']
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
                error_msg = str(e)
                error_type = type(e).__name__
                
                print(f"❌ Erro ao processar arquivo {nome_arquivo}: {error_type} - {error_msg}")
                
                # CORREÇÃO: Mensagem de erro mais específica baseada no tipo de erro
                if 'entry_date' in error_msg.lower() or "'entry_date'" in error_msg:
                    # Tentar normalizar novamente para diagnóstico
                    try:
                        from FunCalculos import _normalize_trades_dataframe
                        df_test = _normalize_trades_dataframe(df.copy())
                        has_entry_date = 'entry_date' in df_test.columns
                        has_pnl = 'pnl' in df_test.columns
                        entry_date_valid = df_test['entry_date'].notna().sum() if has_entry_date else 0
                        pnl_valid = df_test['pnl'].notna().sum() if has_pnl else 0
                        
                        error_msg = (
                            f"Coluna 'entry_date' não encontrada ou inválida. "
                            f"entry_date existe: {has_entry_date}, válidos: {entry_date_valid}, "
                            f"pnl existe: {has_pnl}, válidos: {pnl_valid}. "
                            f"Verifique se o arquivo contém a coluna 'Abertura' com datas válidas."
                        )
                    except Exception as diag_error:
                        error_msg = f"Erro ao processar arquivo: {error_msg}. Diagnóstico adicional falhou: {diag_error}"
                
                resultados_individuais[nome_arquivo] = {
                    "error": error_msg,
                    "error_type": error_type,
                    "info_arquivo": {
                        "nome_arquivo": nome_arquivo,
                        "total_registros": len(df) if 'df' in locals() else 0,
                        "colunas_disponiveis": list(df.columns) if 'df' in locals() and not df.empty else []
                    }
                }
        
        print(f"📋 Resultados individuais processados: {list(resultados_individuais.keys())}")
        
        # Concatenar todos os DataFrames em um só para análise consolidada
        print(f"🔗 Processando dados consolidados:")
        print(f"   📊 Total de registros consolidados: {sum(len(df) for df in dataframes)}")
        
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        print(f"   📋 DataFrame consolidado criado com {len(df_consolidado)} registros")
        print(f"   📅 Colunas consolidadas ANTES da normalização: {list(df_consolidado.columns)[:10]}...")
        
        # CORREÇÃO CRÍTICA: Normalizar o DataFrame consolidado SEMPRE, não apenas se faltar
        # Quando concatenamos DataFrames, eles podem ter colunas diferentes (um tem 'Abertura', outro tem 'entry_date')
        # A normalização garante que todos tenham as mesmas colunas padronizadas
        from FunCalculos import _normalize_trades_dataframe
        print(f"   🔄 Normalizando DataFrame consolidado SEMPRE (shape: {df_consolidado.shape})...")
        print(f"      Colunas antes: {list(df_consolidado.columns)[:15]}...")
        
        df_consolidado = _normalize_trades_dataframe(df_consolidado)
        
        if df_consolidado.empty:
            print(f"   ⚠️ DataFrame consolidado ficou vazio após normalização")
        else:
            entry_date_valid = df_consolidado['entry_date'].notna().sum() if 'entry_date' in df_consolidado.columns else 0
            pnl_valid = df_consolidado['pnl'].notna().sum() if 'pnl' in df_consolidado.columns else 0
            print(f"   ✅ DataFrame consolidado normalizado: entry_date válidos: {entry_date_valid}/{len(df_consolidado)}, pnl válidos: {pnl_valid}/{len(df_consolidado)}")
            print(f"      Colunas depois: {list(df_consolidado.columns)[:15]}...")
        
        # CORREÇÃO CRÍTICA: Aplicar filtros no DataFrame consolidado ANTES de processar
        if filtros:
            print(f"   🔍 Aplicando filtros ao DataFrame consolidado...")
            df_consolidado_antes = len(df_consolidado)
            df_consolidado = aplicar_filtros_basicos(df_consolidado, filtros)
            df_consolidado = df_consolidado.reset_index(drop=True)
            df_consolidado_depois = len(df_consolidado)
            print(f"   ✅ Filtros aplicados ao consolidado: {df_consolidado_antes} -> {df_consolidado_depois} registros")
        
        # CORREÇÃO: Passar taxas customizadas também para o consolidado
        resultado_consolidado = processar_backtest_completo(
            df_consolidado, 
            capital_inicial=capital_inicial, 
            cdi=cdi,
            taxa_corretagem=taxa_corretagem,
            taxa_emolumentos=taxa_emolumentos
        )
        # Padronizar chaves também no consolidado
        try:
            if 'Day of Week Analysis' in resultado_consolidado:
                resultado_consolidado['day_of_week'] = resultado_consolidado['Day of Week Analysis']
            if 'Monthly Analysis' in resultado_consolidado:
                resultado_consolidado['monthly'] = resultado_consolidado['Monthly Analysis']
            if 'Equity Curve Data' in resultado_consolidado:
                resultado_consolidado['equity_curve_data'] = resultado_consolidado['Equity Curve Data']
            if 'Position Sizing' in resultado_consolidado:
                resultado_consolidado['position_sizing'] = resultado_consolidado['Position Sizing']
                resultado_consolidado['positionSizing'] = resultado_consolidado['Position Sizing']
            if 'Trade Duration' in resultado_consolidado:
                resultado_consolidado['trade_duration'] = resultado_consolidado['Trade Duration']
                resultado_consolidado['tradeDuration'] = resultado_consolidado['Trade Duration']
            if 'Operational Costs' in resultado_consolidado:
                resultado_consolidado['operational_costs'] = resultado_consolidado['Operational Costs']
                resultado_consolidado['operationalCosts'] = resultado_consolidado['Operational Costs']
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
        # CORREÇÃO: Normalizar o DataFrame se necessário
        from FunCalculos import _normalize_trades_dataframe
        if 'entry_date' not in df.columns or 'pnl' not in df.columns:
            df = _normalize_trades_dataframe(df)
            if df.empty:
                return []
        
        # Encontrar coluna de resultado (já deve estar normalizada como 'pnl')
        resultado_col = 'pnl' if 'pnl' in df.columns else None
        data_col = 'entry_date' if 'entry_date' in df.columns else None
        
        # Fallback para outras colunas se normalização falhou
        if resultado_col is None:
            for col_name in ['operation_result', 'resultado', 'Res. Intervalo', 'Res. Operação']:
                if col_name in df.columns:
                    resultado_col = col_name
                    break
        
        if data_col is None:
            for col_name in ['data_abertura', 'Abertura', 'data']:
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
        
        # CORREÇÃO CRÍTICA: Normalizar CADA DataFrame individualmente antes de concatenar
        # Isso garante que todos tenham entry_date e pnl no formato correto
        from FunCalculos import _normalize_trades_dataframe
        dataframes_normalizados = []
        for i, df in enumerate(dataframes):
            print(f"🔄 api_tabela: Normalizando DataFrame {i+1}/{len(dataframes)}...")
            df_original_len = len(df)
            df_normalized = _normalize_trades_dataframe(df)
            if df_normalized.empty:
                print(f"   ⚠️ DataFrame {i+1} ficou vazio após normalização (tinha {df_original_len} linhas)")
            else:
                entry_date_valid = df_normalized['entry_date'].notna().sum() if 'entry_date' in df_normalized.columns else 0
                pnl_valid = df_normalized['pnl'].notna().sum() if 'pnl' in df_normalized.columns else 0
                print(f"   ✅ DataFrame {i+1} normalizado: entry_date válidos: {entry_date_valid}/{len(df_normalized)}, pnl válidos: {pnl_valid}/{len(df_normalized)}")
            dataframes_normalizados.append(df_normalized)
        
        # Concatenar todos os DataFrames normalizados em um só
        df_consolidado = pd.concat(dataframes_normalizados, ignore_index=True)
        
        # Validar que temos dados após normalização
        if df_consolidado.empty:
            return jsonify({"error": "Após normalização, todos os arquivos ficaram vazios. Verifique se há dados válidos nas colunas 'Abertura' e de resultado."}), 400
        
        # Validar colunas obrigatórias
        if 'entry_date' not in df_consolidado.columns:
            return jsonify({"error": "Não foi possível criar coluna 'entry_date'. Verifique se o arquivo contém a coluna 'Abertura' com datas válidas."}), 400
        
        if 'pnl' not in df_consolidado.columns:
            return jsonify({"error": "Não foi possível criar coluna 'pnl'. Verifique se o arquivo contém coluna de resultado (Res. Intervalo, Res. Operação, etc.)."}), 400
        
        # CORREÇÃO: Aplicar filtros de período personalizado
        filtros = _parse_filters_from_request(request)
        if filtros:
            df_consolidado = aplicar_filtros_basicos(df_consolidado, filtros)
            df_consolidado = df_consolidado.reset_index(drop=True)
        
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

        # Padronizar chaves adicionais
        if 'Position Sizing' in resultado:
            resultado['position_sizing'] = resultado['Position Sizing']
            resultado['positionSizing'] = resultado['Position Sizing']
        if 'Trade Duration' in resultado:
            resultado['trade_duration'] = resultado['Trade Duration']
            resultado['tradeDuration'] = resultado['Trade Duration']
        if 'Operational Costs' in resultado:
            resultado['operational_costs'] = resultado['Operational Costs']
            resultado['operationalCosts'] = resultado['Operational Costs']
        
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

        # CORREÇÃO: Aplicar filtros de período personalizado
        filtros = _parse_filters_from_request(request)
        if filtros:
            df = aplicar_filtros_basicos(df, filtros)
            df = df.reset_index(drop=True)

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

        # CORREÇÃO: Aplicar filtros de período personalizado
        filtros = _parse_filters_from_request(request)
        if filtros:
            df = aplicar_filtros_basicos(df, filtros)
            df = df.reset_index(drop=True)

        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # CORREÇÃO: Extrair taxas usando função auxiliar
        taxa_corretagem, taxa_emolumentos = _extrair_taxas_do_request(request)
        
        # Usar a função completa com taxas customizadas
        resultado = processar_backtest_completo(
            df, 
            capital_inicial=capital_inicial, 
            cdi=cdi,
            taxa_corretagem=taxa_corretagem,
            taxa_emolumentos=taxa_emolumentos
        )

        if 'Position Sizing' in resultado:
            resultado['position_sizing'] = resultado['Position Sizing']
            resultado['positionSizing'] = resultado['Position Sizing']
        if 'Trade Duration' in resultado:
            resultado['trade_duration'] = resultado['Trade Duration']
            resultado['tradeDuration'] = resultado['Trade Duration']
        if 'Operational Costs' in resultado:
            resultado['operational_costs'] = resultado['Operational Costs']
            resultado['operationalCosts'] = resultado['Operational Costs']
        
        # Adicionar metadados úteis
        resultado["metadata"] = {
            "total_trades": len(df),
            "capital_inicial": capital_inicial,
            "cdi": cdi,
            "filters": filtros,
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
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erro em api_correlacao: {e}")
        print(f"Detalhes: {error_details}")
        return jsonify({"error": f"Erro ao processar correlação: {str(e)}"}), 500


@app.route('/api/hourly-results', methods=['POST'])
def api_hourly_results():
    """
    Endpoint para análise de resultados por hora
    """
    try:
        # Verificar se tem arquivo
        if 'file' not in request.files and 'files' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        dataframes = []
        arquivos_processados = []
        
        # Processar arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_safe(arquivo)
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
        
        # Processar múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_safe(arquivo)
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo válido encontrado"}), 400
        
        # Concatenar DataFrames
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # CORREÇÃO: Normalizar antes de processar
        from FunCalculos import _normalize_trades_dataframe, _detect_pnl_column
        if 'entry_date' not in df_consolidado.columns or 'pnl' not in df_consolidado.columns:
            df_consolidado = _normalize_trades_dataframe(df_consolidado)
            if df_consolidado.empty:
                return jsonify({"error": "Após normalização, o arquivo ficou vazio."}), 400
        
        # Validar colunas necessárias
        if 'entry_date' not in df_consolidado.columns:
            return jsonify({"error": "Coluna obrigatória 'entry_date' não encontrada"}), 400
        
        # CORREÇÃO: Detectar coluna de PnL corretamente
        pnl_col = _detect_pnl_column(df_consolidado)
        if pnl_col is None:
            return jsonify({"error": "Coluna de PnL não encontrada"}), 400
        
        # CORREÇÃO: Aplicar filtros de período personalizado
        filtros = _parse_filters_from_request(request)
        if filtros:
            df_consolidado = aplicar_filtros_basicos(df_consolidado, filtros)
            df_consolidado = df_consolidado.reset_index(drop=True)
        
        # Filtrar dados válidos
        df_valid = df_consolidado.dropna(subset=['entry_date', pnl_col]).copy()
        if df_valid.empty:
            return jsonify({"error": "Nenhum dado válido encontrado após filtros"}), 400
        
        # Garantir que entry_date é datetime
        df_valid['entry_date'] = pd.to_datetime(df_valid['entry_date'], errors='coerce')
        df_valid = df_valid[df_valid['entry_date'].notna()].copy()
        
        if df_valid.empty:
            return jsonify({"error": "Nenhuma data válida encontrada"}), 400
        
        # Extrair hora e minutos da entrada para processar períodos customizados
        df_valid['hour'] = df_valid['entry_date'].dt.hour
        df_valid['minute'] = df_valid['entry_date'].dt.minute
        df_valid['total_minutes'] = df_valid['hour'] * 60 + df_valid['minute']
        
        # Processar períodos customizados se fornecidos
        custom_periods = []
        custom_periods_str = request.form.get('custom_periods')
        if custom_periods_str:
            try:
                import json
                custom_periods = json.loads(custom_periods_str)
            except:
                pass
        
        # Se não há períodos customizados, usar períodos padrão
        if not custom_periods:
            custom_periods = [
                {"start": "09:00", "end": "11:00", "label": "Abertura"},
                {"start": "11:00", "end": "14:00", "label": "Meio-dia"},
                {"start": "14:00", "end": "17:30", "label": "Tarde"},
                {"start": "17:30", "end": "21:00", "label": "Pós-mercado"}
            ]
        
        # Processar cada período customizado
        period_results = []
        for period in custom_periods:
            start_time = period.get('start', '09:00')
            end_time = period.get('end', '11:00')
            label = period.get('label', f"{start_time} - {end_time}")
            
            # Converter horários para minutos
            start_hour, start_min = map(int, start_time.split(':'))
            end_hour, end_min = map(int, end_time.split(':'))
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            
            # Filtrar trades dentro do período
            if start_minutes <= end_minutes:
                # Período normal (não cruza meia-noite)
                period_trades = df_valid[
                    (df_valid['total_minutes'] >= start_minutes) & 
                    (df_valid['total_minutes'] < end_minutes)
                ]
            else:
                # Período que cruza meia-noite
                period_trades = df_valid[
                    (df_valid['total_minutes'] >= start_minutes) | 
                    (df_valid['total_minutes'] < end_minutes)
                ]
            
            # Sempre adicionar o período, mesmo que não tenha trades
            total_trades = len(period_trades)
            
            if total_trades > 0:
                # Calcular métricas do período quando há trades
                total_pnl = float(period_trades[pnl_col].sum())
                winning_trades = period_trades[period_trades[pnl_col] > 0]
                losing_trades = period_trades[period_trades[pnl_col] < 0]
                win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
                
                gross_profit = float(winning_trades[pnl_col].sum()) if len(winning_trades) > 0 else 0.0
                gross_loss = abs(float(losing_trades[pnl_col].sum())) if len(losing_trades) > 0 else 0.0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.99 if gross_profit > 0 else 0.0)
                
                avg_win = round(gross_profit / len(winning_trades), 2) if len(winning_trades) > 0 else 0.0
                avg_loss = round(gross_loss / len(losing_trades), 2) if len(losing_trades) > 0 else 0.0
            else:
                # Período sem trades - valores zerados
                total_pnl = 0.0
                win_rate = 0.0
                profit_factor = 0.0
                avg_win = 0.0
                avg_loss = 0.0
            
            # Sempre adicionar o período à lista de resultados
            period_results.append({
                "period": f"{start_time}-{end_time}",
                "label": label,
                "trades": total_trades,
                "pnl_total": round(total_pnl, 2),
                "win_rate": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "avg_win": avg_win,
                "avg_loss": avg_loss
            })
        
        # Calcular resumo
        total_pnl = sum(r['pnl_total'] for r in period_results)
        # Melhor e pior período apenas entre os que têm trades
        periods_with_trades = [r for r in period_results if r['trades'] > 0]
        best_period = max(periods_with_trades, key=lambda x: x['pnl_total']) if periods_with_trades else None
        worst_period = min(periods_with_trades, key=lambda x: x['pnl_total']) if periods_with_trades else None
        
        # Retornar no formato esperado pelo frontend
        resultado = {
            "summary": {
                "total_periods": len(period_results),  # Total de períodos configurados (incluindo sem trades)
                "total_pnl": round(total_pnl, 2),
                "best_period": best_period,
                "worst_period": worst_period
            },
            "results": period_results,
            "custom_periods": custom_periods,
            "info_arquivos": {
                "total_arquivos": len(arquivos_processados),
                "nomes_arquivos": arquivos_processados,
                "total_registros": len(df_consolidado),
                "registros_apos_filtros": len(df_valid)
            }
        }
        
        return jsonify(make_json_serializable(resultado))
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Erro em api_hourly_results: {e}")
        print(f"Detalhes: {error_details}")
        return jsonify({"error": f"Erro ao processar resultados horários: {str(e)}"}), 500


# Sistema de eventos em memória (pode ser substituído por banco de dados no futuro)
_events_storage = []
_event_id_counter = 1

@app.route('/api/admin/events', methods=['GET', 'POST', 'PUT', 'DELETE'])
def api_admin_events():
    """
    Endpoint para gerenciar eventos administrativos
    GET: Lista eventos (com filtros opcionais)
    POST: Cria novo evento
    PUT: Atualiza evento existente
    DELETE: Remove evento
    """
    global _events_storage, _event_id_counter
    
    try:
        if request.method == 'GET':
            # Listar eventos com filtros opcionais
            event_type = request.args.get('type')
            status = request.args.get('status')
            date_from = request.args.get('date_from')
            date_to = request.args.get('date_to')
            special_only = request.args.get('special_only', 'false').lower() == 'true'
            event_category = request.args.get('event_category')  # Para eventos especiais
            event_date = request.args.get('event_date')  # Data do evento especial
            
            filtered_events = _events_storage.copy()
            
            # Aplicar filtros
            if event_type:
                filtered_events = [e for e in filtered_events if e.get('type') == event_type]
            
            if status:
                filtered_events = [e for e in filtered_events if e.get('status') == status]
            
            # Filtro para eventos especiais
            if special_only:
                filtered_events = [e for e in filtered_events if e.get('is_special', False)]
            
            # Filtro por categoria de evento especial
            if event_category:
                filtered_events = [
                    e for e in filtered_events 
                    if e.get('is_special', False) and 
                    e.get('special_event', {}).get('event_category') == event_category
                ]
            
            # Filtro por data do evento especial
            if event_date:
                try:
                    event_date_dt = pd.to_datetime(event_date).date()
                    filtered_events = [
                        e for e in filtered_events
                        if e.get('is_special', False) and
                        e.get('special_event', {}).get('event_date') and
                        pd.to_datetime(e.get('special_event', {}).get('event_date')).date() == event_date_dt
                    ]
                except:
                    pass
            
            # Filtro por data de criação
            if date_from:
                try:
                    date_from_dt = pd.to_datetime(date_from)
                    filtered_events = [
                        e for e in filtered_events 
                        if pd.to_datetime(e.get('created_at', '2000-01-01')) >= date_from_dt
                    ]
                except:
                    pass
            
            if date_to:
                try:
                    date_to_dt = pd.to_datetime(date_to)
                    filtered_events = [
                        e for e in filtered_events 
                        if pd.to_datetime(e.get('created_at', '2099-12-31')) <= date_to_dt
                    ]
                except:
                    pass
            
            # Ordenar por data (mais recente primeiro) ou por data do evento especial se for especial
            def sort_key(event):
                if event.get('is_special', False) and event.get('special_event', {}).get('event_date'):
                    try:
                        return pd.to_datetime(event.get('special_event', {}).get('event_date'))
                    except:
                        return pd.to_datetime(event.get('created_at', ''))
                return pd.to_datetime(event.get('created_at', ''))
            
            filtered_events.sort(key=sort_key, reverse=True)
            
            return jsonify({
                "events": filtered_events,
                "total": len(filtered_events),
                "special_count": len([e for e in filtered_events if e.get('is_special', False)]),
                "message": "Eventos listados com sucesso"
            })
        
        elif request.method == 'POST':
            # Criar novo evento
            data = request.get_json() if request.is_json else request.form.to_dict()
            
            # Validar campos obrigatórios
            required_fields = ['title', 'type', 'description']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                return jsonify({
                    "error": f"Campos obrigatórios faltando: {', '.join(missing_fields)}"
                }), 400
            
            # Verificar se é evento especial
            is_special = data.get('is_special', False) or data.get('special', False)
            if isinstance(is_special, str):
                is_special = is_special.lower() in ('true', '1', 'yes', 'sim')
            
            # Criar evento base
            new_event = {
                "id": _event_id_counter,
                "title": data.get('title'),
                "type": data.get('type'),  # 'info', 'warning', 'error', 'success', 'maintenance', 'special'
                "description": data.get('description'),
                "status": data.get('status', 'active'),  # 'active', 'resolved', 'archived'
                "priority": data.get('priority', 'medium'),  # 'low', 'medium', 'high', 'critical'
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": data.get('created_by', 'system'),
                "metadata": data.get('metadata', {}),
                "is_special": is_special
            }
            
            # Adicionar campos específicos para eventos especiais
            if is_special:
                # Processar array de datas se fornecido
                dates = data.get('dates', [])
                if dates and isinstance(dates, list) and len(dates) > 0:
                    # Se há múltiplas datas, usar a primeira como event_date e salvar todas em dates
                    new_event["dates"] = dates
                    event_date = dates[0] if dates else (data.get('event_date') or data.get('data_ocorrencia'))
                else:
                    # Se não há array, usar event_date ou date
                    event_date = data.get('event_date') or data.get('date') or data.get('data_ocorrencia')
                    new_event["dates"] = [event_date] if event_date else []
                
                new_event["special_event"] = {
                    "event_date": event_date,  # Data do evento especial (primeira data se múltiplas)
                    "event_category": data.get('event_category') or data.get('categoria', 'market'),  # 'market', 'holiday', 'economic', 'corporate', 'other'
                    "impact": data.get('impact', 'medium'),  # 'low', 'medium', 'high', 'critical'
                    "market_affected": data.get('market_affected', []),  # Lista de mercados afetados
                    "recurring": data.get('recurring', False),  # Se é evento recorrente (ex: feriados)
                    "recurrence_pattern": data.get('recurrence_pattern'),  # 'yearly', 'monthly', 'weekly', etc.
                    "tags": data.get('tags', []),  # Tags para categorização
                    "related_events": data.get('related_events', []),  # IDs de eventos relacionados
                    "notes": data.get('notes', '')  # Notas adicionais sobre o evento
                }
            
            _events_storage.append(new_event)
            _event_id_counter += 1
            
            return jsonify({
                "message": "Evento criado com sucesso",
                "event": new_event
            }), 201
        
        elif request.method == 'PUT':
            # Atualizar evento existente
            data = request.get_json() if request.is_json else request.form.to_dict()
            event_id = data.get('id') or request.args.get('id')
            
            if not event_id:
                return jsonify({"error": "ID do evento é obrigatório"}), 400
            
            try:
                event_id = int(event_id)
            except:
                return jsonify({"error": "ID do evento inválido"}), 400
            
            # Encontrar evento
            event_index = None
            for i, event in enumerate(_events_storage):
                if event.get('id') == event_id:
                    event_index = i
                    break
            
            if event_index is None:
                return jsonify({"error": "Evento não encontrado"}), 404
            
            # Atualizar campos permitidos
            allowed_fields = ['title', 'type', 'description', 'status', 'priority', 'metadata', 'is_special', 'special']
            for field in allowed_fields:
                if field in data:
                    if field == 'special':
                        _events_storage[event_index]['is_special'] = data[field]
                    else:
                        _events_storage[event_index][field] = data[field]
            
            # Atualizar campos de evento especial se for especial
            if data.get('is_special') or data.get('special') or _events_storage[event_index].get('is_special', False):
                if 'special_event' not in _events_storage[event_index]:
                    _events_storage[event_index]['special_event'] = {}
                
                special_fields = [
                    'event_date', 'data_ocorrencia', 'event_category', 'categoria',
                    'impact', 'market_affected', 'recurring', 'recurrence_pattern',
                    'tags', 'related_events', 'notes'
                ]
                
                for field in special_fields:
                    if field in data:
                        if field == 'data_ocorrencia':
                            _events_storage[event_index]['special_event']['event_date'] = data[field]
                        elif field == 'categoria':
                            _events_storage[event_index]['special_event']['event_category'] = data[field]
                        else:
                            _events_storage[event_index]['special_event'][field] = data[field]
            
            _events_storage[event_index]['updated_at'] = datetime.now().isoformat()
            
            return jsonify({
                "message": "Evento atualizado com sucesso",
                "event": _events_storage[event_index]
            })
        
        elif request.method == 'DELETE':
            # Remover evento
            event_id = request.args.get('id') or (request.get_json() if request.is_json else {}).get('id')
            
            if not event_id:
                return jsonify({"error": "ID do evento é obrigatório"}), 400
            
            try:
                event_id = int(event_id)
            except:
                return jsonify({"error": "ID do evento inválido"}), 400
            
            # Encontrar e remover evento
            event_index = None
            for i, event in enumerate(_events_storage):
                if event.get('id') == event_id:
                    event_index = i
                    break
            
            if event_index is None:
                return jsonify({"error": "Evento não encontrado"}), 404
            
            removed_event = _events_storage.pop(event_index)
            
            return jsonify({
                "message": "Evento removido com sucesso",
                "event": removed_event
            })
        
    except Exception as e:
        print(f"❌ Erro ao processar eventos: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar eventos: {str(e)}"}), 500


@app.route('/api/admin/events/special', methods=['GET', 'POST'])
def api_admin_events_special():
    """
    Endpoint específico para eventos especiais
    GET: Lista apenas eventos especiais (com filtros opcionais)
    POST: Cria novo evento especial
    """
    global _events_storage, _event_id_counter
    
    try:
        if request.method == 'GET':
            # Listar apenas eventos especiais com filtros opcionais
            event_category = request.args.get('event_category')
            event_date_from = request.args.get('event_date_from')
            event_date_to = request.args.get('event_date_to')
            impact = request.args.get('impact')
            recurring = request.args.get('recurring')
            
            # Filtrar apenas eventos especiais
            special_events = [e for e in _events_storage if e.get('is_special', False)]
            
            # Aplicar filtros específicos
            if event_category:
                special_events = [
                    e for e in special_events
                    if e.get('special_event', {}).get('event_category') == event_category
                ]
            
            if impact:
                special_events = [
                    e for e in special_events
                    if e.get('special_event', {}).get('impact') == impact
                ]
            
            if recurring:
                recurring_bool = recurring.lower() == 'true'
                special_events = [
                    e for e in special_events
                    if e.get('special_event', {}).get('recurring') == recurring_bool
                ]
            
            # Filtro por data do evento
            if event_date_from:
                try:
                    date_from_dt = pd.to_datetime(event_date_from)
                    special_events = [
                        e for e in special_events
                        if e.get('special_event', {}).get('event_date') and
                        pd.to_datetime(e.get('special_event', {}).get('event_date')) >= date_from_dt
                    ]
                except:
                    pass
            
            if event_date_to:
                try:
                    date_to_dt = pd.to_datetime(event_date_to)
                    special_events = [
                        e for e in special_events
                        if e.get('special_event', {}).get('event_date') and
                        pd.to_datetime(e.get('special_event', {}).get('event_date')) <= date_to_dt
                    ]
                except:
                    pass
            
            # Ordenar por data do evento especial
            def sort_key(event):
                if event.get('special_event', {}).get('event_date'):
                    try:
                        return pd.to_datetime(event.get('special_event', {}).get('event_date'))
                    except:
                        return pd.to_datetime(event.get('created_at', ''))
                return pd.to_datetime(event.get('created_at', ''))
            
            special_events.sort(key=sort_key, reverse=True)
            
            # Normalizar eventos para facilitar uso no frontend
            normalized_events = []
            for event in special_events:
                special_event_data = event.get('special_event', {})
                # Extrair todas as datas do evento (pode estar em dates, date, ou event_date)
                event_dates = event.get('dates', []) or event.get('datas', [])
                if not event_dates:
                    # Se não há array de dates, usar event_date como array com um elemento
                    event_date = special_event_data.get('event_date') or event.get('date')
                    event_dates = [event_date] if event_date else []
                
                normalized_event = {
                    "id": event.get('id'),
                    "title": event.get('title'),
                    "name": event.get('title'),  # Alias para compatibilidade
                    "date": event_dates[0] if event_dates else (special_event_data.get('event_date') or event.get('created_at')),
                    "dates": event_dates,  # Array com todas as datas
                    "event_date": special_event_data.get('event_date') or (event_dates[0] if event_dates else None),
                    "data_ocorrencia": special_event_data.get('event_date') or (event_dates[0] if event_dates else None),
                    "impact": special_event_data.get('impact', 'medium'),
                    "description": event.get('description') or special_event_data.get('notes', ''),
                    "descricao": event.get('description') or special_event_data.get('notes', ''),
                    "event_category": special_event_data.get('event_category', 'market'),
                    "categoria": special_event_data.get('event_category', 'market'),
                    "market_affected": special_event_data.get('market_affected', []),
                    "mercados_afetados": special_event_data.get('market_affected', []),
                    "recurring": special_event_data.get('recurring', False),
                    "recorrente": special_event_data.get('recurring', False),
                    "recurrence_pattern": special_event_data.get('recurrence_pattern'),
                    "padrao_recorrencia": special_event_data.get('recurrence_pattern'),
                    "tags": special_event_data.get('tags', []),
                    "etiquetas": special_event_data.get('tags', []),
                    "created_at": event.get('created_at'),
                    "updated_at": event.get('updated_at'),
                    # Manter estrutura original para compatibilidade
                    "special_event": special_event_data,
                    "is_special": True
                }
                normalized_events.append(normalized_event)
            
            # Agrupar por categoria para estatísticas
            categories = {}
            for event in special_events:
                cat = event.get('special_event', {}).get('event_category', 'other')
                categories[cat] = categories.get(cat, 0) + 1
            
            return jsonify({
                "events": normalized_events,
                "total": len(normalized_events),
                "statistics": {
                    "by_category": categories,
                    "by_impact": {
                        impact: len([e for e in special_events if e.get('special_event', {}).get('impact') == impact])
                        for impact in ['low', 'medium', 'high', 'critical']
                    },
                    "recurring_count": len([e for e in special_events if e.get('special_event', {}).get('recurring', False)])
                },
                "message": "Eventos especiais listados com sucesso"
            })
        
        elif request.method == 'POST':
            # Criar novo evento especial
            data = request.get_json() if request.is_json else request.form.to_dict()
            
            # Validar campos obrigatórios para evento especial
            required_fields = ['title', 'description', 'event_date']
            missing_fields = [
        field for field in required_fields 
        if (field == 'event_date' and not data.get(field) and not data.get('data_ocorrencia')) 
        or (field != 'event_date' and not data.get(field))
    ]
            
            if missing_fields:
                return jsonify({
                    "error": f"Campos obrigatórios faltando: {', '.join(missing_fields)}"
                }), 400
            
            # Criar evento especial
            new_event = {
                "id": _event_id_counter,
                "title": data.get('title'),
                "type": data.get('type', 'special'),
                "description": data.get('description'),
                "status": data.get('status', 'active'),
                "priority": data.get('priority', 'medium'),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "created_by": data.get('created_by', 'system'),
                "metadata": data.get('metadata', {}),
                "is_special": True,
                "special_event": {
                    "event_date": data.get('event_date') or data.get('data_ocorrencia'),
                    "event_category": data.get('event_category') or data.get('categoria', 'market'),
                    "impact": data.get('impact', 'medium'),
                    "market_affected": data.get('market_affected', []),
                    "recurring": data.get('recurring', False),
                    "recurrence_pattern": data.get('recurrence_pattern'),
                    "tags": data.get('tags', []),
                    "related_events": data.get('related_events', []),
                    "notes": data.get('notes', '')
                }
            }
            
            _events_storage.append(new_event)
            _event_id_counter += 1
            
            return jsonify({
                "message": "Evento especial criado com sucesso",
                "event": new_event
            }), 201
        
    except Exception as e:
        print(f"❌ Erro ao processar eventos especiais: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar eventos especiais: {str(e)}"}), 500


@app.route('/api/admin/events/upcoming-special', methods=['GET'])
def api_admin_events_upcoming_special():
    """
    Endpoint para listar eventos especiais futuros (próximos eventos)
    """
    global _events_storage
    
    try:
        days_ahead = int(request.args.get('days', 30))  # Próximos 30 dias por padrão
        
        now = datetime.now()
        future_date = now + timedelta(days=days_ahead)
        
        # Filtrar eventos especiais futuros
        upcoming_events = []
        for event in _events_storage:
            if not event.get('is_special', False):
                continue
            
            event_date_str = event.get('special_event', {}).get('event_date')
            if not event_date_str:
                continue
            
            try:
                event_date = pd.to_datetime(event_date_str)
                if now <= event_date <= future_date and event.get('status') == 'active':
                    upcoming_events.append(event)
            except:
                continue
        
        # Ordenar por data do evento
        upcoming_events.sort(key=lambda x: pd.to_datetime(x.get('special_event', {}).get('event_date', '')))
        
        return jsonify({
            "events": upcoming_events,
            "total": len(upcoming_events),
            "days_ahead": days_ahead,
            "message": f"Próximos {len(upcoming_events)} eventos especiais nos próximos {days_ahead} dias"
        })
        
    except Exception as e:
        print(f"❌ Erro ao buscar eventos especiais futuros: {e}")
        return jsonify({"error": f"Erro ao buscar eventos especiais futuros: {str(e)}"}), 500


# ============ SISTEMA DE MAILING ============
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def _send_email(to_email: str, subject: str, body: str, html_body: str = None, attachments: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Função auxiliar para enviar emails
    Retorna dict com status da operação
    """
    try:
        # Configurações de email a partir de variáveis de ambiente
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME', '')
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        from_email = os.getenv('SMTP_FROM_EMAIL', smtp_username)
        
        # Se não tem configurações, retornar erro
        if not smtp_username or not smtp_password:
            return {
                "success": False,
                "error": "Configurações de email não encontradas. Configure SMTP_USERNAME e SMTP_PASSWORD nas variáveis de ambiente."
            }
        
        # Criar mensagem
        msg = MIMEMultipart('alternative')
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Adicionar corpo do email
        if html_body:
            part1 = MIMEText(body, 'plain')
            part2 = MIMEText(html_body, 'html')
            msg.attach(part1)
            msg.attach(part2)
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # Adicionar anexos se houver
        if attachments:
            for attachment in attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment['content'])
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {attachment["filename"]}'
                )
                msg.attach(part)
        
        # Conectar e enviar
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        
        return {
            "success": True,
            "message": f"Email enviado com sucesso para {to_email}"
        }
    
    except Exception as e:
        print(f"❌ Erro ao enviar email: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/api/mailing/send', methods=['POST'])
def api_mailing_send():
    """
    Endpoint para enviar emails
    POST: Envia email para destinatário(s)
    """
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Validar campos obrigatórios
        to_email = data.get('to') or data.get('email') or data.get('to_email')
        subject = data.get('subject') or data.get('titulo')
        body = data.get('body') or data.get('message') or data.get('mensagem')
        
        if not to_email:
            return jsonify({"error": "Campo 'to' (destinatário) é obrigatório"}), 400
        
        if not subject:
            return jsonify({"error": "Campo 'subject' (assunto) é obrigatório"}), 400
        
        if not body:
            return jsonify({"error": "Campo 'body' (corpo do email) é obrigatório"}), 400
        
        # Processar múltiplos destinatários (separados por vírgula)
        recipients = [email.strip() for email in str(to_email).split(',')]
        
        # HTML body opcional
        html_body = data.get('html_body') or data.get('html')
        
        # Anexos opcionais (lista de objetos com 'filename' e 'content' em base64)
        attachments = data.get('attachments', [])
        processed_attachments = []
        
        if attachments:
            import base64
            for att in attachments:
                if isinstance(att, dict) and 'filename' in att and 'content' in att:
                    try:
                        # Decodificar base64 se necessário
                        content = att['content']
                        if isinstance(content, str):
                            content = base64.b64decode(content)
                        processed_attachments.append({
                            'filename': att['filename'],
                            'content': content
                        })
                    except Exception as e:
                        print(f"⚠️ Erro ao processar anexo {att.get('filename', 'unknown')}: {e}")
        
        # Enviar email para cada destinatário
        results = []
        for recipient in recipients:
            result = _send_email(
                to_email=recipient,
                subject=subject,
                body=body,
                html_body=html_body,
                attachments=processed_attachments if processed_attachments else None
            )
            results.append({
                "recipient": recipient,
                "success": result.get("success", False),
                "message": result.get("message"),
                "error": result.get("error")
            })
        
        # Verificar se todos foram enviados com sucesso
        all_success = all(r["success"] for r in results)
        
        return jsonify({
            "success": all_success,
            "results": results,
            "total_sent": sum(1 for r in results if r["success"]),
            "total_failed": sum(1 for r in results if not r["success"])
        }), 200 if all_success else 207  # 207 Multi-Status se alguns falharam
    
    except Exception as e:
        print(f"❌ Erro ao processar envio de email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar envio de email: {str(e)}"}), 500

@app.route('/api/mailing/test', methods=['POST'])
def api_mailing_test():
    """
    Endpoint para testar configuração de email
    POST: Envia email de teste
    """
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Email de teste (ou usar o fornecido)
        test_email = data.get('email') or data.get('to') or os.getenv('SMTP_TEST_EMAIL', '')
        
        if not test_email:
            return jsonify({
                "error": "Email de teste não fornecido. Envie 'email' no body da requisição ou configure SMTP_TEST_EMAIL nas variáveis de ambiente."
            }), 400
        
        # Enviar email de teste
        result = _send_email(
            to_email=test_email,
            subject="Teste de Email - DevHub Trader",
            body="Este é um email de teste do sistema DevHub Trader.\n\nSe você recebeu este email, a configuração de email está funcionando corretamente.",
            html_body="""
            <html>
                <body>
                    <h2>Teste de Email - DevHub Trader</h2>
                    <p>Este é um email de teste do sistema DevHub Trader.</p>
                    <p>Se você recebeu este email, a configuração de email está funcionando corretamente.</p>
                    <hr>
                    <p><small>Enviado automaticamente pelo sistema DevHub Trader</small></p>
                </body>
            </html>
            """
        )
        
        if result.get("success"):
            return jsonify({
                "success": True,
                "message": "Email de teste enviado com sucesso",
                "details": result
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Erro desconhecido ao enviar email"),
                "details": result
            }), 500
    
    except Exception as e:
        print(f"❌ Erro ao processar teste de email: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro ao processar teste de email: {str(e)}"}), 500

@app.route('/api/mailing/config', methods=['GET'])
def api_mailing_config():
    """
    Endpoint para verificar configuração de email (sem expor senhas)
    GET: Retorna status da configuração de email
    """
    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', '587'))
        smtp_username = os.getenv('SMTP_USERNAME', '')
        smtp_from_email = os.getenv('SMTP_FROM_EMAIL', '')
        
        # Verificar se está configurado (sem expor senha)
        is_configured = bool(smtp_username and os.getenv('SMTP_PASSWORD', ''))
        
        return jsonify({
            "configured": is_configured,
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "smtp_username": smtp_username if smtp_username else None,
            "smtp_from_email": smtp_from_email if smtp_from_email else None,
            "message": "Configuração de email verificada (senha não exposta)"
        })
    
    except Exception as e:
        return jsonify({"error": f"Erro ao verificar configuração: {str(e)}"}), 500


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
        # CORREÇÃO: Extrair taxas usando função auxiliar (com defaults)
        taxa_corretagem, taxa_emolumentos = _extrair_taxas_do_request(request)
        if taxa_corretagem is None:
            taxa_corretagem = 0.5  # Default: R$ 0,50 por roda
        if taxa_emolumentos is None:
            taxa_emolumentos = 0.03  # Default: R$ 0,03 por roda
        
        # Lista para armazenar todos os DataFrames
        dataframes = []
        arquivos_processados: List[str] = []
        filtros = _parse_filters_from_request(request)
        
        # Verificar se tem arquivo único
        if 'file' in request.files:
            arquivo = request.files['file']
            if arquivo.filename != '':
                df = carregar_csv_trades(arquivo)
                df['source_file'] = arquivo.filename
                dataframes.append(df)
                arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem múltiplos arquivos
        if 'files' in request.files:
            arquivos = request.files.getlist('files')
            for arquivo in arquivos:
                if arquivo.filename != '':
                    df = carregar_csv_trades(arquivo)
                    df['source_file'] = arquivo.filename
                    dataframes.append(df)
                    arquivos_processados.append(arquivo.filename)
        
        # Verificar se tem caminho de arquivo
        if 'path' in request.form:
            path = request.form['path']
            if not os.path.exists(path):
                return jsonify({"error": "Arquivo não encontrado"}), 404
            df = carregar_csv_trades(path)
            df['source_file'] = os.path.basename(path)
            dataframes.append(df)
            arquivos_processados.append(os.path.basename(path))
        
        # Se não tem nenhum arquivo
        if not dataframes:
            return jsonify({"error": "Nenhum arquivo enviado. Use 'file' para um arquivo ou 'files' para múltiplos"}), 400
        
        # Concatenar todos os DataFrames em um só
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        
        # CORREÇÃO: Normalizar ANTES de aplicar filtros para garantir colunas corretas
        from FunCalculos import _normalize_trades_dataframe
        print(f"🔄 Normalizando DataFrame consolidado antes de aplicar filtros...")
        df_consolidado = _normalize_trades_dataframe(df_consolidado)
        
        if df_consolidado.empty:
            return jsonify({"error": "Após normalização, todos os arquivos ficaram vazios."}), 400

        # CORREÇÃO: Aplicar filtros APÓS normalização
        if filtros:
            print(f"🔍 Aplicando filtros ao DataFrame consolidado (shape antes: {df_consolidado.shape})...")
            df_consolidado = aplicar_filtros_basicos(df_consolidado, filtros)
            print(f"✅ Filtros aplicados (shape depois: {df_consolidado.shape})")

        df_consolidado = df_consolidado.reset_index(drop=True)

        arquivo_para_indices = {}
        if 'source_file' in df_consolidado.columns:
            arquivo_para_indices = {
                idx: df_consolidado.at[idx, 'source_file']
                for idx in df_consolidado.index
            }
        
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

        trades_por_arquivo = {}
        if 'source_file' in df_consolidado.columns:
            trades_por_arquivo = (
                df_consolidado['source_file']
                .fillna('Desconhecido')
                .value_counts()
                .to_dict()
            )

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
                "available_strategies": available_strategies,
                "current": filtros
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
                    "trades_por_arquivo": trades_por_arquivo,
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

        filtros = _parse_filters_from_request(request)
        if filtros:
            df = aplicar_filtros_basicos(df, filtros).reset_index(drop=True)

        # Calcular apenas estatísticas essenciais
        estatisticas_gerais = calcular_estatisticas_gerais(df)
        custos = calcular_custos_operacionais(df)
        
        resultado = {
            "summary": estatisticas_gerais,
            "costs": custos,
            "total_records": len(df),
            "filters": filtros
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

        filtros = _parse_filters_from_request(request)
        if filtros:
            df = aplicar_filtros_basicos(df, filtros).reset_index(drop=True)

        # Parâmetros opcionais
        capital_inicial = float(request.form.get('capital_inicial', 100000))
        cdi = float(request.form.get('cdi', 0.12))
        
        # Usar FunCalculos.py para garantir consistência
        from FunCalculos import processar_backtest_completo
        
        # CORREÇÃO: Extrair taxas usando função auxiliar
        taxa_corretagem, taxa_emolumentos = _extrair_taxas_do_request(request)
        
        # Processar backtest completo usando FunCalculos.py
        resultado = processar_backtest_completo(
            df, 
            capital_inicial=capital_inicial, 
            cdi=cdi,
            taxa_corretagem=taxa_corretagem,
            taxa_emolumentos=taxa_emolumentos
        )
        
        # Extrair apenas as métricas principais do resultado
        performance_metrics = resultado.get("Performance Metrics", {})
        
        # CORREÇÃO: Extrair custos operacionais corretamente
        operational_costs = resultado.get("Operational Costs", {})
        
        # Garantir que temos um dict válido
        if not isinstance(operational_costs, dict):
            operational_costs = {}
        
        # Extrair corretagem e emolumentos separadamente
        corretagem_total = float(operational_costs.get("corretagem", 0.0))
        emolumentos_total = float(operational_costs.get("emolumentos", 0.0))
        
        # Se não encontrou nos custos operacionais, buscar nas métricas de performance
        if corretagem_total == 0.0:
            corretagem_total = float(performance_metrics.get("Total Brokerage", performance_metrics.get("Corretagem Total", 0.0)))
        if emolumentos_total == 0.0:
            emolumentos_total = float(performance_metrics.get("Total Fees", performance_metrics.get("Emolumentos Totais", 0.0)))
        
        # Garantir que os valores são números válidos
        if not isinstance(corretagem_total, (int, float)) or pd.isna(corretagem_total):
            corretagem_total = 0.0
        if not isinstance(emolumentos_total, (int, float)) or pd.isna(emolumentos_total):
            emolumentos_total = 0.0
        
        print(f"🔍 api_daily_metrics: Custos operacionais extraídos:")
        print(f"   📊 Operational Costs keys: {list(operational_costs.keys())}")
        print(f"   💼 Corretagem: R$ {corretagem_total:.2f}")
        print(f"   💼 Emolumentos: R$ {emolumentos_total:.2f}")
        print(f"   💼 Total: R$ {corretagem_total + emolumentos_total:.2f}")
        
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
            "corretagem_total": round(corretagem_total, 2),
            "emolumentos_total": round(emolumentos_total, 2),
            "custo_total_operacional": round(corretagem_total + emolumentos_total, 2),
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
            },
            # CORREÇÃO: Adicionar seção de custos operacionais separada
            "custos_operacionais": {
                "corretagem": round(corretagem_total, 2),
                "emolumentos": round(emolumentos_total, 2),
                "total": round(corretagem_total + emolumentos_total, 2),
                "taxa_corretagem_aplicada": taxa_corretagem if taxa_corretagem is not None else None,
                "taxa_emolumentos_aplicada": taxa_emolumentos if taxa_emolumentos is not None else None
            }
        }
        
        if not metricas:
            return jsonify({"error": "Não foi possível calcular métricas"}), 400

        metricas["filters"] = filtros

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
    
    # CORREÇÃO: Normalizar o DataFrame se necessário
    from FunCalculos import _normalize_trades_dataframe
    if 'entry_date' not in df.columns or 'pnl' not in df.columns:
        df = _normalize_trades_dataframe(df)
        if df.empty:
            return {}
    
    print("🔍 DEBUG - Verificação de padronização do drawdown:")
    
    # CORREÇÃO: Validar que temos as colunas necessárias ANTES de usar
    if 'entry_date' not in df.columns:
        print("❌ Coluna 'entry_date' não encontrada após normalização")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        return {}
    
    if 'pnl' not in df.columns:
        print("❌ Coluna 'pnl' não encontrada após normalização")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        return {}
    
    # Verificar se há valores válidos
    entry_date_valid = df['entry_date'].notna().sum()
    pnl_valid = df['pnl'].notna().sum()
    
    if entry_date_valid == 0:
        print(f"⚠️ Nenhuma data válida encontrada (todas são NaT)")
        print(f"   Tentando continuar sem validação de data...")
    
    if pnl_valid == 0:
        print(f"⚠️ Nenhum PnL válido encontrado")
        return {}
    
    # Método 1: FunCalculos.py (trades individuais)
    # Filtrar apenas linhas que têm AMBOS os valores válidos
    df_valid = df[df['entry_date'].notna() & df['pnl'].notna()].copy()
    
    if df_valid.empty:
        print("⚠️ Nenhuma linha com entry_date e pnl válidos")
        return {}
    
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
    
    # CORREÇÃO: Normalizar o DataFrame se necessário
    from FunCalculos import _normalize_trades_dataframe
    if 'entry_date' not in df.columns or 'pnl' not in df.columns:
        df = _normalize_trades_dataframe(df)
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
    
    # Calcular equity curve trade por trade com baseline zero (padronizado)
    pnl_series = df_valid['pnl'].fillna(0).astype(float)
    equity = pnl_series.cumsum()
    equity_with_start = pd.concat([pd.Series([0.0]), equity], ignore_index=True)
    peak = equity_with_start.cummax()
    drawdown = equity_with_start - peak
    
    # Remover o ponto inicial artificial para análises
    equity = equity_with_start.iloc[1:]
    peak = peak.iloc[1:]
    drawdown = drawdown.iloc[1:]
    
    # Drawdown máximo (valor positivo)
    max_drawdown = float(abs(drawdown.min())) if not drawdown.empty else 0.0
    
    # Saldo final
    saldo_final = float(equity.iloc[-1]) if not equity.empty else 0.0
    
    # Capital inicial estimado: maior pico observado (considerando baseline 0)
    peak_max = float(peak.max()) if not peak.empty else 0.0
    capital_inicial = peak_max if peak_max > 0 else max_drawdown
    
    # Percentual do drawdown (baseado no capital inicial)
    max_drawdown_pct = (max_drawdown / capital_inicial * 100) if capital_inicial not in (0, np.nan) else 0.0
    
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

# ============ ROTAS DE CONFIGURAÇÃO DE COMISSÕES ============

@app.route('/api/user/commission-settings', methods=['GET'])
@require_auth
def get_commission_settings():
    """
    Buscar as configurações de comissão do usuário logado
    CORREÇÃO: Separa corretagem e emolumentos
    Retorna defaults se não existir configuração salva
    """
    try:
        user_id = request.user_id
        
        if not supabase_client:
            print(f"[ERROR] Supabase client não inicializado. SUPABASE_URL: {bool(SUPABASE_URL)}, SUPABASE_KEY: {bool(SUPABASE_KEY)}")
            return jsonify({
                "error": "Supabase não configurado. Configure SUPABASE_URL e SUPABASE_KEY nas variáveis de ambiente."
            }), 500
        
        print(f"[DEBUG] Buscando configurações para user_id: {user_id}")
        
        # Buscar configurações do banco
        try:
            # Verificar se a tabela existe tentando fazer uma query simples
            response = supabase_client.table('user_commission_settings')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            print(f"[DEBUG] Resposta do Supabase: {len(response.data) if response.data else 0} registros encontrados")
        except Exception as db_error:
            error_msg = str(db_error)
            print(f"[ERROR] Erro ao buscar no Supabase: {error_msg}")
            import traceback
            traceback.print_exc()
            
            # Se o erro for sobre tabela não encontrada, retornar defaults
            if 'relation' in error_msg.lower() and 'does not exist' in error_msg.lower():
                print(f"[WARN] Tabela user_commission_settings não existe. Retornando defaults.")
                return jsonify({
                    "corretagem": {
                        "method": "fixed_per_roda",
                        "value": 0.50,
                        "overrideExisting": True
                    },
                    "emolumentos": {
                        "method": "fixed_per_roda",
                        "value": 0.03,
                        "overrideExisting": True
                    },
                    "applyDifferenceToPnl": True,
                    "configs": []
                }), 200
            
            return jsonify({
                "error": f"Erro ao buscar configurações no banco: {error_msg}",
                "details": error_msg
            }), 500
        
        # Se não encontrou, retornar defaults separados
        if not response.data or len(response.data) == 0:
            return jsonify({
                # Configurações de corretagem
                "corretagem": {
                    "method": "fixed_per_roda",  # "fixed_per_roda" ou "fixed_per_trade"
                    "value": 0.50,  # R$ 0,50 por roda (padrão mercado brasileiro)
                    "overrideExisting": True
                },
                # Configurações de emolumentos
                "emolumentos": {
                    "method": "fixed_per_roda",  # "fixed_per_roda" ou "percentage"
                    "value": 0.03,  # R$ 0,03 por roda (padrão mercado brasileiro)
                    "overrideExisting": True
                },
                "applyDifferenceToPnl": True,
                "configs": []  # Configurações por ativo
            }), 200
        
        # Retornar configurações encontradas (compatibilidade com formato antigo)
        data = response.data[0]
        
        asset_configs_count = len(data.get('asset_configs', [])) if isinstance(data.get('asset_configs'), list) else 0
        print(f"[DEBUG] Encontradas {asset_configs_count} configurações de ativo no banco")
        if asset_configs_count > 0:
            print(f"[DEBUG] Primeira asset_config do banco: {data.get('asset_configs', [])[0]}")
            if asset_configs_count > 1:
                print(f"[DEBUG] Última asset_config do banco: {data.get('asset_configs', [])[-1]}")
        
        # Se tem formato antigo, converter para novo formato
        if 'corretagem' not in data and 'emolumentos' not in data:
            # Formato antigo - converter
            return jsonify({
                "corretagem": {
                    "method": data.get('corretagem_method', 'fixed_per_roda'),
                    "value": float(data.get('corretagem_value', 0.50)),
                    "overrideExisting": data.get('corretagem_override_existing', True)
                },
                "emolumentos": {
                    "method": data.get('emolumentos_method', 'fixed_per_roda'),
                    "value": float(data.get('emolumentos_value', 0.03)),
                    "overrideExisting": data.get('emolumentos_override_existing', True)
                },
                "applyDifferenceToPnl": data.get('apply_difference_to_pnl', True),
                "configs": data.get('asset_configs', [])
            }), 200
        
        # Formato novo
        return jsonify({
            "corretagem": data.get('corretagem', {
                "method": "fixed_per_roda",
                "value": 0.50,
                "overrideExisting": True
            }),
            "emolumentos": data.get('emolumentos', {
                "method": "fixed_per_roda",
                "value": 0.03,
                "overrideExisting": True
            }),
            "applyDifferenceToPnl": data.get('apply_difference_to_pnl', True),
            "configs": data.get('asset_configs', [])
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Erro ao buscar configurações de comissão: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erro ao buscar configurações: {str(e)}",
            "details": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/api/user/commission-settings', methods=['PUT'])
@require_auth
def save_commission_settings():
    """
    Salvar/atualizar as configurações de comissão do usuário logado
    CORREÇÃO: Separa corretagem e emolumentos
    """
    try:
        user_id = request.user_id
        
        if not supabase_client:
            return jsonify({
                "error": "Supabase não configurado. Configure SUPABASE_URL e SUPABASE_KEY nas variáveis de ambiente."
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Body da requisição vazio"}), 400
        
        # CORREÇÃO: Extrair configurações separadas de corretagem e emolumentos
        corretagem_config = data.get('corretagem', {})
        emolumentos_config = data.get('emolumentos', {})
        
        # Compatibilidade com formato antigo
        if not corretagem_config and not emolumentos_config:
            # Formato antigo - converter
            default_method = data.get('defaultMethod', 'fixed')
            default_value = data.get('defaultValue', 0)
            corretagem_config = {
                "method": default_method,
                "value": default_value,
                "overrideExisting": data.get('overrideExisting', True)
            }
            emolumentos_config = {
                "method": default_method,
                "value": default_value,
                "overrideExisting": data.get('overrideExisting', True)
            }
        
        # Validações para corretagem
        corretagem_method = corretagem_config.get('method', 'fixed_per_roda')
        corretagem_value = corretagem_config.get('value', 0.50)
        if corretagem_method not in ['fixed_per_roda', 'fixed_per_trade']:
            return jsonify({"error": "corretagem.method deve ser 'fixed_per_roda' ou 'fixed_per_trade'"}), 400
        if not isinstance(corretagem_value, (int, float)) or corretagem_value < 0:
            return jsonify({"error": "corretagem.value deve ser um número >= 0"}), 400
        
        # Validações para emolumentos
        emolumentos_method = emolumentos_config.get('method', 'fixed_per_roda')
        emolumentos_value = emolumentos_config.get('value', 0.03)
        if emolumentos_method not in ['fixed_per_roda', 'percentage']:
            return jsonify({"error": "emolumentos.method deve ser 'fixed_per_roda' ou 'percentage'"}), 400
        if not isinstance(emolumentos_value, (int, float)) or emolumentos_value < 0:
            return jsonify({"error": "emolumentos.value deve ser um número >= 0"}), 400
        
        apply_difference_to_pnl = data.get('applyDifferenceToPnl', True)
        configs = data.get('configs', [])
        
        print(f"[DEBUG] Recebido {len(configs) if isinstance(configs, list) else 0} configurações de ativo")
        if isinstance(configs, list) and len(configs) > 0:
            print(f"[DEBUG] Primeira config: {configs[0]}")
            if len(configs) > 1:
                print(f"[DEBUG] Última config: {configs[-1]}")
        
        if not isinstance(configs, list):
            return jsonify({"error": "configs deve ser um array"}), 400
        
        # Validar cada configuração de ativo
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                return jsonify({"error": f"configs[{i}] deve ser um objeto"}), 400
            
            asset = config.get('asset')
            if not asset or not isinstance(asset, str) or asset.strip() == '':
                return jsonify({"error": f"configs[{i}].asset deve ser uma string não vazia"}), 400
            
            # Validar corretagem do ativo
            if 'corretagem' in config:
                if config['corretagem'].get('method') not in ['fixed_per_roda', 'fixed_per_trade']:
                    return jsonify({"error": f"configs[{i}].corretagem.method deve ser 'fixed_per_roda' ou 'fixed_per_trade'"}), 400
                if not isinstance(config['corretagem'].get('value'), (int, float)) or config['corretagem'].get('value') < 0:
                    return jsonify({"error": f"configs[{i}].corretagem.value deve ser um número >= 0"}), 400
            
            # Validar emolumentos do ativo
            if 'emolumentos' in config:
                if config['emolumentos'].get('method') not in ['fixed_per_roda', 'percentage']:
                    return jsonify({"error": f"configs[{i}].emolumentos.method deve ser 'fixed_per_roda' ou 'percentage'"}), 400
                if not isinstance(config['emolumentos'].get('value'), (int, float)) or config['emolumentos'].get('value') < 0:
                    return jsonify({"error": f"configs[{i}].emolumentos.value deve ser um número >= 0"}), 400
        
        # Preparar dados para salvar
        asset_configs = []
        for config in configs:
            asset_config = {
                "asset": config['asset'].upper().strip(),
                "corretagem": config.get('corretagem', corretagem_config),
                "emolumentos": config.get('emolumentos', emolumentos_config)
            }
            asset_configs.append(asset_config)
        
        print(f"[DEBUG] Preparando para salvar {len(asset_configs)} configurações de ativo")
        if len(asset_configs) > 0:
            print(f"[DEBUG] Primeira asset_config: {asset_configs[0]}")
            if len(asset_configs) > 1:
                print(f"[DEBUG] Última asset_config: {asset_configs[-1]}")
        
        # Salvar no banco (upsert)
        upsert_data = {
            "user_id": user_id,
            "corretagem": {
                "method": corretagem_method,
                "value": float(corretagem_value),
                "override_existing": bool(corretagem_config.get('overrideExisting', True))
            },
            "emolumentos": {
                "method": emolumentos_method,
                "value": float(emolumentos_value),
                "override_existing": bool(emolumentos_config.get('overrideExisting', True))
            },
            "apply_difference_to_pnl": bool(apply_difference_to_pnl),
            "asset_configs": asset_configs
        }
        
        # Tentar salvar usando o cliente Supabase
        # Se estiver usando SERVICE_ROLE_KEY, bypassa RLS automaticamente
        try:
            response = supabase_client.table('user_commission_settings')\
                .upsert(upsert_data, on_conflict='user_id')\
                .execute()
            
            if not response.data:
                return jsonify({"error": "Erro ao salvar configurações"}), 500
        except Exception as db_error:
            error_str = str(db_error)
            # Verificar se é erro de RLS
            if 'row-level security' in error_str.lower() or '42501' in error_str:
                print(f"[ERROR] Erro de RLS ao salvar configurações: {db_error}")
                print(f"[ERROR] Isso geralmente acontece quando não está usando SERVICE_ROLE_KEY")
                return jsonify({
                    "error": "Erro ao salvar configurações: violação de política de segurança (RLS). Configure SUPABASE_SERVICE_ROLE_KEY no backend para bypassar RLS.",
                    "details": str(db_error)
                }), 500
            # Re-raise outros erros
            raise
        
        # Verificar o que foi salvo
        saved_data = response.data[0] if response.data else {}
        saved_asset_configs = saved_data.get('asset_configs', [])
        print(f"[DEBUG] Dados salvos no banco: {len(saved_asset_configs) if isinstance(saved_asset_configs, list) else 0} configurações")
        if isinstance(saved_asset_configs, list) and len(saved_asset_configs) > 0:
            print(f"[DEBUG] Primeira config salva: {saved_asset_configs[0]}")
            if len(saved_asset_configs) > 1:
                print(f"[DEBUG] Última config salva: {saved_asset_configs[-1]}")
        
        return jsonify({
            "success": True,
            "message": "Configurações salvas com sucesso",
            "settings": upsert_data
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Erro ao salvar configurações de comissão: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erro ao salvar configurações: {str(e)}"}), 500

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
        traceback.print_exc()