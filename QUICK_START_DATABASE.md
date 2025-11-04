# 🚀 Quick Start - Banco de Dados

**Sistema de Persistência de Dados - DevHub Trader**

---

## ⚡ Setup Rápido (2 minutos)

### 1️⃣ Inicializar Banco
```bash
python setup_database.py
```

### 2️⃣ Testar Sistema
```bash
python test_database.py
```

### 3️⃣ Ativar no Sistema
Edite `.env`:
```env
USE_DATABASE=true
DB_TYPE=sqlite
```

### 4️⃣ Iniciar Aplicação
```bash
python main.py
```

**Pronto! Dados agora são persistentes! 🎉**

---

## 🔧 Modos de Operação

### Modo 1: Memória (Padrão - Desenvolvimento)
```env
USE_DATABASE=false
```
✅ Rápido para testes  
⚠️ Dados perdidos ao reiniciar

### Modo 2: SQLite (Recomendado - Produção Pequena)
```env
USE_DATABASE=true
DB_TYPE=sqlite
DATABASE_URL=devhubtrader.db
```
✅ Persistência de dados  
✅ Fácil configuração  
✅ Zero dependências  
✅ Até ~100 usuários simultâneos

### Modo 3: PostgreSQL (Produção Grande)
```env
USE_DATABASE=true
DB_TYPE=postgresql
DATABASE_URL=postgresql://user:pass@localhost/devhubtrader
```
✅ Alta performance  
✅ Milhares de usuários  
✅ Replicação e backup  
✅ Escalável

---

## 📦 O Que Foi Criado

### Arquivos Principais
1. **`database_schema.sql`** - Schema PostgreSQL completo
2. **`database.py`** - Managers e conexão
3. **`db_integration.py`** - Wrappers de compatibilidade
4. **`setup_database.py`** - Script de inicialização
5. **`test_database.py`** - Testes automáticos
6. **`.env.example`** - Template de configuração

### Funcionalidades
- ✅ **Usuários**: Cadastro, planos, limites
- ✅ **Eventos**: CRUD completo
- ✅ **Análises**: Salvar backtests
- ✅ **Diário Quântico**: Entradas diárias
- ✅ **Portfolios**: Gestão completa
- ✅ **Custos**: Por ativo/usuário

---

## 🎯 Como Usar no Código

### Importar Services
```python
from db_integration import UserService, EventService, AnalysisService, DiaryService
```

### Gerenciar Usuários
```python
# Verificar plano
plan = UserService.get_user_plan('user123')

# Verificar uso
usage = UserService.get_user_usage('user123')

# Verificar limite antes de consumir
check = UserService.check_limit('user123', 'analyses', 5)
if check['allowed']:
    # Consumir recurso
    UserService.increment_usage('user123', 'analyses', 5)
```

### Gerenciar Eventos
```python
# Criar evento
event_id = EventService.create_event(
    user_id='admin',
    date='2024-01-15',
    name='FOMC Meeting',
    event_type='economic',
    impact='high'
)

# Listar eventos
events = EventService.list_events()

# Deletar evento
EventService.delete_event(event_id)
```

### Salvar Análises
```python
# Salvar
analysis_id = AnalysisService.save_analysis(
    user_id='user123',
    title='Backtest WDO',
    analysis_type='backtest',
    data={'total_trades': 150, 'win_rate': 58.3}
)

# Listar
analyses = AnalysisService.get_analyses('user123', 'backtest')

# Deletar
AnalysisService.delete_analysis(analysis_id, 'user123')
```

### Diário Quântico
```python
# Salvar entrada
entry_id = DiaryService.save_entry(
    user_id='user123',
    entry_date='2024-01-15',
    trades_data={'trades': 5, 'pnl': 250.50},
    emotional_state='disciplinado'
)

# Buscar entrada
entry = DiaryService.get_entry('user123', '2024-01-15')

# Listar entradas
entries = DiaryService.get_entries('user123')
```

---

## 🔍 Troubleshooting

### Problema: Erros de importação
**Solução:** O sistema funciona mesmo sem banco ativado (usa memória)
```python
USE_DATABASE=false  # Volta para modo memória
```

### Problema: "Table already exists"
**Solução:** Normal, banco já foi inicializado. Pode ignorar.

### Problema: Dados não persistem
**Solução:** Verifique `.env`:
```bash
cat .env | grep USE_DATABASE
# Deve mostrar: USE_DATABASE=true
```

### Problema: Performance lenta
**Soluções:**
1. Usar PostgreSQL ao invés de SQLite
2. Ativar índices (já criados automaticamente)
3. Limpar logs antigos periodicamente

---

## 📊 Estrutura de Dados

### Users
```python
{
    'id': 'uuid',
    'email': 'user@example.com',
    'name': 'João Trader',
    'plan': 'QUANT_PRO',
    'tokens_used': 450,
    'portfolios_created': 2,
    'analyses_run': 25
}
```

### Special Events
```python
{
    'id': 'uuid',
    'event_date': '2024-01-15',
    'name': 'FOMC Meeting',
    'description': 'Fed meeting',
    'event_type': 'economic',
    'impact': 'high'
}
```

### Saved Analyses
```python
{
    'id': 'uuid',
    'user_id': 'user123',
    'title': 'Backtest WDO',
    'analysis_type': 'backtest',
    'data': {...}  # JSON com todas as métricas
}
```

### Quantum Diary
```python
{
    'id': 'uuid',
    'user_id': 'user123',
    'entry_date': '2024-01-15',
    'trades_data': {...},
    'performance_metrics': {...},
    'emotional_state': 'disciplinado',
    'notes': 'Dia produtivo'
}
```

---

## 🎁 Recursos Extras

### Backup Automático (SQLite)
```bash
# Criar backup diário
cp devhubtrader.db backups/db_$(date +%Y%m%d).db
```

### Ver Conteúdo do Banco
```bash
# SQLite
sqlite3 devhubtrader.db "SELECT * FROM users;"

# PostgreSQL
psql devhubtrader -c "SELECT * FROM users;"
```

### Reset de Dados (Desenvolvimento)
```bash
# Deletar banco e recriar
rm devhubtrader.db
python setup_database.py
```

---

## ⚙️ Variáveis de Ambiente

### Mínimas (SQLite)
```env
USE_DATABASE=true
DB_TYPE=sqlite
DATABASE_URL=devhubtrader.db
```

### Completas (PostgreSQL)
```env
USE_DATABASE=true
DB_TYPE=postgresql
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Pool de conexões (opcional)
DB_MIN_CONNECTIONS=1
DB_MAX_CONNECTIONS=10
```

---

## 📝 Checklist de Implementação

### Desenvolvimento
- [x] Criar arquivos de banco de dados
- [x] Implementar managers
- [x] Criar wrappers de compatibilidade
- [x] Escrever testes
- [ ] Executar `python setup_database.py`
- [ ] Executar `python test_database.py`
- [ ] Configurar `.env`

### Produção
- [ ] Escolher PostgreSQL ou SQLite
- [ ] Executar schema SQL
- [ ] Configurar backup automático
- [ ] Configurar monitoramento
- [ ] Testar em ambiente de staging
- [ ] Migrar dados de produção
- [ ] Deploy!

---

## 🎯 Próximos Passos

1. **Executar Setup**
   ```bash
   python setup_database.py
   ```

2. **Executar Testes**
   ```bash
   python test_database.py
   ```

3. **Configurar `.env`**
   ```bash
   cp .env.example .env
   # Edite conforme necessário
   ```

4. **Iniciar Sistema**
   ```bash
   python main.py
   ```

**Sistema com banco de dados está pronto! 🚀**

---

## 💡 Dicas

- ✅ Use SQLite para começar (mais simples)
- ✅ Migre para PostgreSQL quando escalar
- ✅ Faça backups regulares
- ✅ Monitore uso de recursos
- ✅ Limpe logs antigos mensalmente

**Boa sorte com seu sistema profissional de trading! 📈💰**

