"""
Database Manager - DevHub Trader
Gerenciamento de banco de dados com PostgreSQL/SQLite
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Suporte para PostgreSQL ou SQLite
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # 'postgresql' ou 'sqlite'

if DB_TYPE == 'postgresql':
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    DB_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/devhubtrader')
else:
    import sqlite3
    DB_URL = os.getenv('DATABASE_URL', 'devhubtrader.db')

# ============================================
# CLASSE DE CONEXÃO
# ============================================

class Database:
    """Gerenciador de conexão com banco de dados"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or DB_URL
        self.db_type = DB_TYPE
        
    def get_connection(self):
        """Retorna conexão apropriada"""
        if self.db_type == 'postgresql':
            return psycopg2.connect(self.db_url)
        else:
            conn = sqlite3.connect(self.db_url)
            conn.row_factory = sqlite3.Row
            return conn
    
    def execute(self, query: str, params: tuple = None, fetch: bool = False):
        """Executa query e opcionalmente retorna resultados"""
        conn = self.get_connection()
        try:
            if self.db_type == 'postgresql':
                cur = conn.cursor(cursor_factory=RealDictCursor)
            else:
                cur = conn.cursor()
            
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            
            if fetch:
                results = cur.fetchall()
                if self.db_type == 'sqlite':
                    # Converter Row para dict
                    results = [dict(row) for row in results]
                return results
            else:
                conn.commit()
                return cur.lastrowid if self.db_type == 'sqlite' else None
        finally:
            conn.close()
    
    def init_database(self):
        """Inicializa banco de dados com schema básico (SQLite)"""
        if self.db_type == 'sqlite':
            schema = """
            -- Tabela de usuários
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                plan TEXT DEFAULT 'FREE_FOREVER',
                tokens_used INTEGER DEFAULT 0,
                portfolios_created INTEGER DEFAULT 0,
                analyses_run INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            
            -- Tabela de eventos especiais
            CREATE TABLE IF NOT EXISTS special_events (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                event_date DATE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                event_type TEXT DEFAULT 'economic',
                impact TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            
            -- Tabela de análises salvas
            CREATE TABLE IF NOT EXISTS saved_analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                analysis_type TEXT NOT NULL,
                file_name TEXT,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            
            -- Tabela de portfolios
            CREATE TABLE IF NOT EXISTS portfolios (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                initial_capital REAL NOT NULL,
                current_capital REAL,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            
            -- Tabela de diário quântico
            CREATE TABLE IF NOT EXISTS quantum_diary (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                entry_date DATE NOT NULL,
                trades_data TEXT NOT NULL,
                performance_metrics TEXT,
                emotional_state TEXT,
                discipline_score REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, entry_date)
            );
            
            -- Tabela de custos por ativo
            CREATE TABLE IF NOT EXISTS asset_costs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                broker_fee REAL NOT NULL,
                exchange_fee REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, symbol)
            );
            
            -- Índices
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_events_date ON special_events(event_date);
            CREATE INDEX IF NOT EXISTS idx_analyses_user ON saved_analyses(user_id);
            CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id);
            CREATE INDEX IF NOT EXISTS idx_diary_user_date ON quantum_diary(user_id, entry_date);
            """
            
            conn = self.get_connection()
            try:
                conn.executescript(schema)
                conn.commit()
                print("✅ Banco de dados SQLite inicializado com sucesso!")
            finally:
                conn.close()

# ============================================
# FUNÇÕES DE GERENCIAMENTO DE USUÁRIOS
# ============================================

class UserManager:
    """Gerenciador de usuários"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_user(self, email: str, name: str = None, plan: str = 'FREE_FOREVER') -> str:
        """Cria novo usuário"""
        import uuid
        user_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO users (id, email, name, plan, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        
        self.db.execute(query, (user_id, email, name or email.split('@')[0], plan, datetime.now()))
        return user_id
    
    def get_user(self, user_id: str = None, email: str = None) -> Optional[Dict]:
        """Busca usuário por ID ou email"""
        if user_id:
            query = "SELECT * FROM users WHERE id = ?"
            params = (user_id,)
        elif email:
            query = "SELECT * FROM users WHERE email = ?"
            params = (email,)
        else:
            return None
        
        results = self.db.execute(query, params, fetch=True)
        return results[0] if results else None
    
    def update_user(self, user_id: str, **kwargs) -> bool:
        """Atualiza dados do usuário"""
        allowed_fields = ['name', 'plan', 'tokens_used', 'portfolios_created', 'analyses_run', 'last_login']
        
        fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
        
        values.append(datetime.now())
        values.append(user_id)
        
        query = f"""
            UPDATE users 
            SET {', '.join(fields)}, updated_at = ?
            WHERE id = ?
        """
        
        self.db.execute(query, tuple(values))
        return True
    
    def increment_usage(self, user_id: str, resource: str, amount: int = 1) -> bool:
        """Incrementa uso de recurso"""
        resource_map = {
            'tokens': 'tokens_used',
            'portfolios': 'portfolios_created',
            'analyses': 'analyses_run'
        }
        
        field = resource_map.get(resource)
        if not field:
            return False
        
        query = f"""
            UPDATE users 
            SET {field} = {field} + ?, updated_at = ?
            WHERE id = ?
        """
        
        self.db.execute(query, (amount, datetime.now(), user_id))
        return True
    
    def check_limit(self, user_id: str, resource: str, amount: int = 1) -> Dict[str, Any]:
        """Verifica se usuário pode consumir recurso"""
        user = self.get_user(user_id)
        if not user:
            return {"allowed": False, "reason": "Usuário não encontrado"}
        
        # Limites por plano
        limits = {
            'FREE_FOREVER': {'tokens': 100, 'portfolios': 0, 'analyses': 5},
            'QUANT_STARTER': {'tokens': 500, 'portfolios': 0, 'analyses': 20},
            'QUANT_PRO': {'tokens': 5000, 'portfolios': 5, 'analyses': -1},
            'QUANT_MASTER': {'tokens': -1, 'portfolios': -1, 'analyses': -1}
        }
        
        plan = user.get('plan', 'FREE_FOREVER')
        plan_limits = limits.get(plan, limits['FREE_FOREVER'])
        
        limit = plan_limits.get(resource, 0)
        
        # Ilimitado
        if limit == -1:
            return {"allowed": True, "remaining": -1}
        
        # Verificar uso atual
        usage_map = {
            'tokens': user.get('tokens_used', 0),
            'portfolios': user.get('portfolios_created', 0),
            'analyses': user.get('analyses_run', 0)
        }
        
        current = usage_map.get(resource, 0)
        remaining = max(0, limit - current)
        
        if amount > remaining:
            return {
                "allowed": False,
                "reason": f"Limite de {resource} excedido",
                "current": current,
                "limit": limit,
                "remaining": remaining,
                "requested": amount
            }
        
        return {
            "allowed": True,
            "current": current,
            "limit": limit,
            "remaining": remaining - amount
        }

# ============================================
# FUNÇÕES DE GERENCIAMENTO DE EVENTOS
# ============================================

class EventManager:
    """Gerenciador de eventos especiais"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_event(self, user_id: str, date: str, name: str, 
                    description: str = None, event_type: str = 'economic', 
                    impact: str = 'medium') -> str:
        """Cria novo evento"""
        import uuid
        event_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO special_events (id, user_id, event_date, name, description, event_type, impact)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db.execute(query, (event_id, user_id, date, name, description, event_type, impact))
        return event_id
    
    def get_events(self, user_id: str = None, date: str = None) -> List[Dict]:
        """Lista eventos (opcionalmente filtrando por usuário ou data)"""
        query = "SELECT * FROM special_events WHERE 1=1"
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if date:
            query += " AND event_date = ?"
            params.append(date)
        
        query += " ORDER BY event_date DESC"
        
        return self.db.execute(query, tuple(params) if params else None, fetch=True)
    
    def delete_event(self, event_id: str) -> bool:
        """Deleta evento"""
        query = "DELETE FROM special_events WHERE id = ?"
        self.db.execute(query, (event_id,))
        return True

# ============================================
# FUNÇÕES DE GERENCIAMENTO DE ANÁLISES
# ============================================

class AnalysisManager:
    """Gerenciador de análises salvas"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def save_analysis(self, user_id: str, title: str, analysis_type: str, 
                     data: Dict, description: str = None, file_name: str = None) -> str:
        """Salva nova análise"""
        import uuid
        analysis_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO saved_analyses (id, user_id, title, description, analysis_type, file_name, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        data_json = json.dumps(data)
        self.db.execute(query, (analysis_id, user_id, title, description, analysis_type, file_name, data_json))
        return analysis_id
    
    def get_analyses(self, user_id: str, analysis_type: str = None) -> List[Dict]:
        """Lista análises do usuário"""
        query = "SELECT * FROM saved_analyses WHERE user_id = ?"
        params = [user_id]
        
        if analysis_type:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        
        query += " ORDER BY created_at DESC"
        
        results = self.db.execute(query, tuple(params), fetch=True)
        
        # Desserializar JSON
        for result in results:
            if 'data' in result and isinstance(result['data'], str):
                result['data'] = json.loads(result['data'])
        
        return results
    
    def delete_analysis(self, analysis_id: str, user_id: str) -> bool:
        """Deleta análise (apenas do próprio usuário)"""
        query = "DELETE FROM saved_analyses WHERE id = ? AND user_id = ?"
        self.db.execute(query, (analysis_id, user_id))
        return True

# ============================================
# FUNÇÕES DE GERENCIAMENTO DE PORTFOLIOS
# ============================================

class PortfolioManager:
    """Gerenciador de portfolios"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_portfolio(self, user_id: str, name: str, initial_capital: float,
                        description: str = None) -> str:
        """Cria novo portfolio"""
        import uuid
        portfolio_id = str(uuid.uuid4())
        
        query = """
            INSERT INTO portfolios (id, user_id, name, description, initial_capital, current_capital, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        self.db.execute(query, (portfolio_id, user_id, name, description, initial_capital, initial_capital, 'active'))
        return portfolio_id
    
    def get_portfolios(self, user_id: str, status: str = None) -> List[Dict]:
        """Lista portfolios do usuário"""
        query = "SELECT * FROM portfolios WHERE user_id = ?"
        params = [user_id]
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC"
        
        return self.db.execute(query, tuple(params), fetch=True)
    
    def update_portfolio(self, portfolio_id: str, **kwargs) -> bool:
        """Atualiza portfolio"""
        allowed_fields = ['name', 'description', 'current_capital', 'status']
        
        fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in allowed_fields:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
        
        values.append(datetime.now())
        values.append(portfolio_id)
        
        query = f"""
            UPDATE portfolios 
            SET {', '.join(fields)}, updated_at = ?
            WHERE id = ?
        """
        
        self.db.execute(query, tuple(values))
        return True

# ============================================
# FUNÇÕES DE GERENCIAMENTO DE DIÁRIO QUÂNTICO
# ============================================

class DiaryManager:
    """Gerenciador do Diário Quântico"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def __getattribute__(self, name):
        """Evitar erro de referência circular"""
        return object.__getattribute__(self, name)
    
    def save_entry(self, user_id: str, entry_date: str, trades_data: Dict,
                  performance_metrics: Dict = None, emotional_state: str = None,
                  notes: str = None) -> str:
        """Salva entrada do diário"""
        import uuid
        entry_id = str(uuid.uuid4())
        
        # Verificar se já existe entrada para esta data
        existing = self.get_entry(user_id, entry_date)
        
        if existing:
            # Atualizar existente
            query = """
                UPDATE quantum_diary
                SET trades_data = ?, performance_metrics = ?, emotional_state = ?, notes = ?, updated_at = ?
                WHERE user_id = ? AND entry_date = ?
            """
            self.db.execute(query, (
                json.dumps(trades_data),
                json.dumps(performance_metrics) if performance_metrics else None,
                emotional_state,
                notes,
                datetime.now(),
                user_id,
                entry_date
            ))
            return existing['id']
        else:
            # Criar nova
            query = """
                INSERT INTO quantum_diary (id, user_id, entry_date, trades_data, performance_metrics, emotional_state, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.db.execute(query, (
                entry_id,
                user_id,
                entry_date,
                json.dumps(trades_data),
                json.dumps(performance_metrics) if performance_metrics else None,
                emotional_state,
                notes
            ))
            return entry_id
    
    def get_entry(self, user_id: str, entry_date: str) -> Optional[Dict]:
        """Busca entrada específica"""
        query = "SELECT * FROM quantum_diary WHERE user_id = ? AND entry_date = ?"
        results = self.db.execute(query, (user_id, entry_date), fetch=True)
        
        if results:
            entry = results[0]
            # Desserializar JSON
            if 'trades_data' in entry:
                entry['trades_data'] = json.loads(entry['trades_data'])
            if 'performance_metrics' in entry and entry['performance_metrics']:
                entry['performance_metrics'] = json.loads(entry['performance_metrics'])
            return entry
        return None
    
    def get_entries(self, user_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Lista entradas do diário"""
        query = "SELECT * FROM quantum_diary WHERE user_id = ?"
        params = [user_id]
        
        if start_date:
            query += " AND entry_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY entry_date DESC"
        
        results = self.db.execute(query, tuple(params), fetch=True)
        
        # Desserializar JSON
        for entry in results:
            if 'trades_data' in entry:
                entry['trades_data'] = json.loads(entry['trades_data'])
            if 'performance_metrics' in entry and entry['performance_metrics']:
                entry['performance_metrics'] = json.loads(entry['performance_metrics'])
        
        return results

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

def migrate_memory_to_db():
    """Migra dados em memória para banco de dados"""
    from main import _ADMIN_EVENTS, _ADMIN_USER_PLANS, _USER_TOKEN_USAGE
    
    db = Database()
    db.init_database()
    
    user_mgr = UserManager(db)
    event_mgr = EventManager(db)
    
    print("🔄 Migrando dados em memória para banco de dados...")
    
    # Migrar usuários
    for user_id, plan in _ADMIN_USER_PLANS.items():
        try:
            # Buscar uso
            usage = _USER_TOKEN_USAGE.get(user_id, {})
            
            # Criar ou atualizar usuário
            existing = user_mgr.get_user(user_id=user_id)
            if not existing:
                user_mgr.create_user(
                    email=f"{user_id}@temp.com",
                    name=user_id,
                    plan=plan
                )
            
            # Atualizar uso
            user_mgr.update_user(
                user_id=user_id,
                tokens_used=usage.get('tokens_used', 0),
                portfolios_created=usage.get('portfolios_created', 0),
                analyses_run=usage.get('analyses_run', 0)
            )
            print(f"  ✅ Usuário {user_id} migrado")
        except Exception as e:
            print(f"  ❌ Erro ao migrar usuário {user_id}: {e}")
    
    # Migrar eventos
    for event_id, event in _ADMIN_EVENTS.items():
        try:
            event_mgr.create_event(
                user_id='admin',  # Eventos do admin
                date=event.get('date'),
                name=event.get('name'),
                description=event.get('description', ''),
                event_type=event.get('type', 'economic'),
                impact=event.get('impact', 'medium')
            )
            print(f"  ✅ Evento {event.get('name')} migrado")
        except Exception as e:
            print(f"  ❌ Erro ao migrar evento {event_id}: {e}")
    
    print("✅ Migração concluída!")

# ============================================
# INSTÂNCIA GLOBAL
# ============================================

# Criar instância global do banco
db = Database()

# Criar managers
user_manager = UserManager(db)
event_manager = EventManager(db)
analysis_manager = AnalysisManager(db)
portfolio_manager = PortfolioManager(db)
diary_manager = DiaryManager(db)

# ============================================
# INICIALIZAÇÃO
# ============================================

def init_db():
    """Inicializa banco de dados"""
    db.init_database()
    print("✅ Banco de dados pronto para uso!")

if __name__ == '__main__':
    # Inicializar e migrar dados
    init_db()
    print("\n🔄 Deseja migrar dados em memória para o banco? (s/n)")
    # migrate_memory_to_db()  # Descomentar para migrar

