# 🗄️ Guia de Configuração do Banco de Dados

**DevHub Trader - Database Setup Guide**

---

## 🎯 Visão Geral

O sistema agora suporta **persistência de dados** através de banco de dados, substituindo os estados em memória temporários.

**Suporte:**
- ✅ SQLite (desenvolvimento/pequena escala)
- ✅ PostgreSQL (produção/alta performance)

---

## 📦 Instalação de Dependências

### Para SQLite (padrão)
```bash
# SQLite já vem com Python, não precisa instalar nada!
pip install -r requirements.txt
```

### Para PostgreSQL
```bash
pip install psycopg2-binary
```

Adicione ao `requirements.txt`:
```
psycopg2-binary==2.9.9
```

---

## 🚀 Quick Start (SQLite)

### 1. Executar Setup
```bash
python setup_database.py
```

### 2. Ativar Banco de Dados
Crie/edite o arquivo `.env`:
```env
USE_DATABASE=true
DB_TYPE=sqlite
DATABASE_URL=devhubtrader.db
```

### 3. Iniciar Aplicação
```bash
python main.py
```

**Pronto!** O sistema agora usa banco de dados SQLite.

---

## 🐘 Setup PostgreSQL (Produção)

### 1. Criar Banco de Dados
```bash
# Criar banco
createdb devhubtrader

# Ou via psql
psql -U postgres
CREATE DATABASE devhubtrader;
\q
```

### 2. Executar Schema
```bash
psql -U postgres -d devhubtrader -f database_schema.sql
```

### 3. Configurar Variáveis de Ambiente
```env
USE_DATABASE=true
DB_TYPE=postgresql
DATABASE_URL=postgresql://usuario:senha@localhost/devhubtrader
```

### 4. Iniciar Aplicação
```bash
python main.py
```

---

## 🔄 Migração de Dados

Se você já tem dados em memória (usuários, eventos, análises), pode migrá-los:

### Automático
```bash
python setup_database.py
# Responda 's' quando perguntado sobre migração
```

### Manual
```python
from database import migrate_memory_to_db
migrate_memory_to_db()
```

---

## 📊 Estrutura do Banco de Dados

### Tabelas Principais

| Tabela | Descrição |
|--------|-----------|
| `users` | Usuários e seus planos |
| `user_profiles` | Dados adicionais e preferências |
| `special_events` | Eventos especiais (FOMC, CPI, etc) |
| `saved_analyses` | Análises de backtest salvas |
| `quantum_diary` | Entradas do diário quântico |
| `portfolios` | Portfolios gerenciados |
| `portfolio_strategies` | Estratégias dentro dos portfolios |
| `portfolio_trades` | Trades de cada portfolio |
| `asset_costs` | Custos personalizados por ativo |
| `usage_logs` | Logs de uso do sistema |

### Views

| View | Descrição |
|------|-----------|
| `user_usage_summary` | Resumo de uso por usuário |
| `portfolio_performance` | Performance de cada portfolio |

---

## 🔧 Modo Híbrido (Desenvolvimento)

O sistema suporta **modo híbrido**:
- `USE_DATABASE=false` → Usa memória (padrão, mais rápido para dev)
- `USE_DATABASE=true` → Usa banco de dados (persistente)

**Vantagem:** Você pode desenvolver com memória e ativar DB só em produção!

---

## 📝 Exemplos de Uso

### Criar Usuário
```python
from database import user_manager

user_id = user_manager.create_user(
    email='trader@example.com',
    name='João Trader',
    plan='QUANT_PRO'
)
```

### Salvar Evento
```python
from database import event_manager

event_id = event_manager.create_event(
    user_id='user123',
    date='2024-01-15',
    name='FOMC Meeting',
    description='Federal Reserve interest rate decision',
    event_type='economic',
    impact='high'
)
```

### Salvar Análise
```python
from database import analysis_manager

analysis_id = analysis_manager.save_analysis(
    user_id='user123',
    title='Backtest WDO Janeiro 2024',
    analysis_type='backtest',
    data={
        'total_trades': 150,
        'win_rate': 58.3,
        'profit_factor': 1.85
    },
    file_name='wdo_jan_2024.csv'
)
```

### Salvar Entrada do Diário
```python
from database import diary_manager

entry_id = diary_manager.save_entry(
    user_id='user123',
    entry_date='2024-01-15',
    trades_data={
        'trades': 5,
        'pnl': 250.50
    },
    emotional_state='disciplinado',
    notes='Dia produtivo, segui o plano'
)
```

---

## 🔍 Queries Úteis

### Ver Todos os Usuários
```sql
SELECT * FROM user_usage_summary;
```

### Ver Performance de Portfolios
```sql
SELECT * FROM portfolio_performance WHERE status = 'active';
```

### Ver Eventos Próximos
```sql
SELECT * FROM special_events 
WHERE event_date >= CURRENT_DATE 
ORDER BY event_date;
```

### Ver Uso por Usuário
```sql
SELECT email, plan, tokens_used, analyses_run 
FROM users 
ORDER BY tokens_used DESC;
```

---

## 🛠️ Manutenção

### Backup (SQLite)
```bash
# Backup simples
cp devhubtrader.db devhubtrader_backup_$(date +%Y%m%d).db

# Ou usando sqlite3
sqlite3 devhubtrader.db ".backup 'backup.db'"
```

### Backup (PostgreSQL)
```bash
pg_dump devhubtrader > backup_$(date +%Y%m%d).sql
```

### Limpar Uso Mensal (Reset de Tokens)
```sql
-- PostgreSQL
SELECT reset_monthly_usage();

-- SQLite
UPDATE users SET tokens_used = 0, analyses_run = 0;
```

### Ver Estatísticas de Uso
```sql
SELECT 
    plan,
    COUNT(*) as total_users,
    AVG(tokens_used) as avg_tokens,
    AVG(analyses_run) as avg_analyses
FROM users
GROUP BY plan;
```

---

## ⚙️ Configurações Avançadas

### Arquivo `.env` Completo
```env
# Banco de Dados
USE_DATABASE=true
DB_TYPE=postgresql
DATABASE_URL=postgresql://user:password@localhost:5432/devhubtrader

# ou para SQLite:
# DB_TYPE=sqlite
# DATABASE_URL=devhubtrader.db

# Servidor
FLASK_ENV=production
PORT=5002

# OpenAI
OPENAI_API_KEY=sk-...

# Outros
MAX_UPLOAD_SIZE=16777216
```

### Pool de Conexões (PostgreSQL)
Para alta performance, configure pool de conexões:

```python
from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)
```

---

## 🚨 Troubleshooting

### Erro: "table already exists"
**Solução:** Banco já foi inicializado. Ignore o erro ou delete o DB e recrie.

### Erro: "connection refused"
**Solução:** PostgreSQL não está rodando. Inicie o serviço:
```bash
# Linux
sudo service postgresql start

# macOS
brew services start postgresql

# Docker
docker start postgres_container
```

### Erro: "permission denied"
**Solução:** Ajuste permissões do usuário no PostgreSQL:
```sql
GRANT ALL PRIVILEGES ON DATABASE devhubtrader TO seu_usuario;
```

### Performance Lenta
**Soluções:**
1. Criar índices adicionais
2. Usar pool de conexões
3. Ativar VACUUM (PostgreSQL)
4. Otimizar queries com EXPLAIN

---

## 📊 Monitoramento

### Ver Tamanho do Banco
```sql
-- PostgreSQL
SELECT pg_size_pretty(pg_database_size('devhubtrader'));

-- SQLite
SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size();
```

### Ver Tabelas Maiores
```sql
-- PostgreSQL
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## 🎯 Próximos Passos

Após configurar o banco de dados:

1. ✅ Sistema usa persistência real
2. ✅ Dados não são perdidos ao reiniciar
3. ✅ Suporte a múltiplos usuários simultâneos
4. ✅ Logs de auditoria
5. ✅ Backup e recuperação

**Seu sistema está pronto para produção!** 🚀

