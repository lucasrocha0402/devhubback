# 🗄️ Sistema de Banco de Dados - DevHub Trader

## 🎯 O QUE FOI IMPLEMENTADO

Sistema completo de persistência de dados com suporte a **SQLite** e **PostgreSQL**.

---

## 📦 ARQUIVOS CRIADOS

### 1. `database_schema.sql` (PostgreSQL)
**O que contém:**
- ✅ 11 tabelas completas
- ✅ Views para análises
- ✅ Triggers automáticos
- ✅ Funções auxiliares
- ✅ Índices otimizados
- ✅ Validações e constraints

**Tabelas:**
- `users` - Usuários e planos
- `user_profiles` - Perfis completos
- `special_events` - Eventos especiais
- `saved_analyses` - Análises salvas
- `quantum_diary` - Diário quântico
- `portfolios` - Gestão de portfolios
- `portfolio_strategies` - Estratégias
- `portfolio_trades` - Trades
- `asset_costs` - Custos por ativo
- `api_keys` - Chaves de API
- `usage_logs` - Logs de uso

### 2. `database.py` (Python)
**O que contém:**
- ✅ Classe `Database` para conexão
- ✅ `UserManager` - Gerenciamento de usuários
- ✅ `EventManager` - Eventos especiais
- ✅ `AnalysisManager` - Análises salvas
- ✅ `PortfolioManager` - Portfolios
- ✅ `DiaryManager` - Diário quântico
- ✅ Função de migração de dados
- ✅ Suporte a SQLite e PostgreSQL

### 3. `db_integration.py` (Python)
**O que contém:**
- ✅ Wrappers de compatibilidade
- ✅ `UserService` - Abstração de usuários
- ✅ `EventService` - Abstração de eventos
- ✅ `AnalysisService` - Abstração de análises
- ✅ `DiaryService` - Abstração de diário
- ✅ Modo híbrido (DB ou memória)

### 4. `setup_database.py` (Script)
**O que faz:**
- ✅ Inicializa banco de dados
- ✅ Cria todas as tabelas
- ✅ Oferece migração de dados
- ✅ Mostra instruções de configuração

### 5. `DATABASE_SETUP_GUIDE.md` (Guia)
**O que contém:**
- ✅ Instruções completas de setup
- ✅ Exemplos de uso
- ✅ Queries úteis
- ✅ Troubleshooting
- ✅ Manutenção e backup

### 6. `.env.example` (Template)
**O que contém:**
- ✅ Todas as variáveis de ambiente necessárias
- ✅ Configurações de banco de dados
- ✅ Configurações de servidor
- ✅ Configurações de email
- ✅ Features e segurança

---

## 🚀 COMO USAR

### Opção 1: SQLite (Mais Simples)
```bash
# 1. Executar setup
python setup_database.py

# 2. Ativar no .env
echo "USE_DATABASE=true" >> .env
echo "DB_TYPE=sqlite" >> .env

# 3. Iniciar aplicação
python main.py
```

### Opção 2: PostgreSQL (Produção)
```bash
# 1. Criar banco
createdb devhubtrader

# 2. Executar schema
psql -d devhubtrader -f database_schema.sql

# 3. Configurar .env
cp .env.example .env
# Editar .env com suas configurações

# 4. Iniciar aplicação
python main.py
```

---

## 🔄 MODO HÍBRIDO

O sistema suporta **dois modos simultâneos**:

### Modo Memória (Desenvolvimento)
```env
USE_DATABASE=false
```
- ✅ Mais rápido para testes
- ✅ Não precisa configurar nada
- ⚠️ Dados perdidos ao reiniciar

### Modo Database (Produção)
```env
USE_DATABASE=true
```
- ✅ Dados persistentes
- ✅ Suporta múltiplos usuários
- ✅ Backup e recuperação
- ✅ Pronto para escalar

**Você pode alternar entre os modos apenas mudando a variável!**

---

## 📊 FUNCIONALIDADES DO BANCO

### ✅ Gerenciamento de Usuários
- Criar, atualizar, buscar usuários
- Controlar planos e limites
- Rastrear uso de recursos
- Histórico de login

### ✅ Eventos Especiais
- CRUD completo
- Filtrar por data/tipo
- Integração com análises
- Indicadores de impacto

### ✅ Análises Salvas
- Salvar backtests
- Organizar por tipo
- Buscar histórico
- Compartilhar análises

### ✅ Diário Quântico
- Entradas diárias
- Métricas de performance
- Estado emocional
- Notas e reflexões

### ✅ Portfolio Manager
- Múltiplos portfolios
- Estratégias por portfolio
- Trades organizados
- Performance tracking

### ✅ Custos Personalizados
- Por ativo
- Por usuário
- Corretagem e taxas
- Fácil atualização

---

## 🎁 BONUS: Funções Úteis

### Verificar Limite
```python
from db_integration import UserService

check = UserService.check_limit('user123', 'analyses', 5)
if check['allowed']:
    print(f"✅ Pode consumir. Restam: {check['remaining']}")
else:
    print(f"❌ Limite excedido: {check['reason']}")
```

### Incrementar Uso
```python
from db_integration import UserService

UserService.increment_usage('user123', 'tokens', 10)
```

### Salvar Evento
```python
from db_integration import EventService

event_id = EventService.create_event(
    user_id='admin',
    date='2024-01-15',
    name='FOMC Meeting',
    event_type='economic',
    impact='high'
)
```

---

## 🎊 BENEFÍCIOS

### Para Desenvolvimento
- ✅ Modo memória para testes rápidos
- ✅ Fácil reset de dados
- ✅ Não precisa configurar DB

### Para Produção
- ✅ Dados persistentes e seguros
- ✅ Escalável para milhares de usuários
- ✅ Backup e recuperação
- ✅ Auditoria completa
- ✅ Performance otimizada

### Para o Negócio
- ✅ Sistema profissional
- ✅ Pronto para escalar
- ✅ Conformidade com LGPD
- ✅ Multi-tenant ready

---

## ✅ CHECKLIST DE IMPLEMENTAÇÃO

### Backend (Python)
- [x] Schema SQL (PostgreSQL)
- [x] Models e Managers (database.py)
- [x] Wrappers de integração (db_integration.py)
- [x] Script de setup (setup_database.py)
- [x] Migração de dados em memória
- [x] Suporte a SQLite e PostgreSQL
- [x] Modo híbrido (DB ou memória)

### Documentação
- [x] Guia de setup completo
- [x] Exemplos de uso
- [x] Template de .env
- [x] Troubleshooting
- [x] README detalhado

### Funcionalidades
- [x] Usuários e planos
- [x] Eventos especiais
- [x] Análises salvas
- [x] Diário quântico
- [x] Portfolio manager
- [x] Custos personalizados
- [x] Logs de uso
- [x] API keys

---

## 🚀 PRÓXIMO PASSO

**Execute o setup:**
```bash
python setup_database.py
```

**E configure o .env:**
```bash
cp .env.example .env
# Edite .env conforme necessário
```

**Pronto! Seu sistema tem banco de dados profissional!** 🎉

