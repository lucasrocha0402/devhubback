-- ============================================
-- DEVHUB TRADER - Database Schema
-- Banco de dados para persistência de dados
-- ============================================

-- Extensões necessárias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- TABELA: users (sincronizada com Supabase Auth)
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Dados do plano
    plan VARCHAR(50) DEFAULT 'FREE_FOREVER',
    plan_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Uso de recursos
    tokens_used INTEGER DEFAULT 0,
    portfolios_created INTEGER DEFAULT 0,
    analyses_run INTEGER DEFAULT 0,
    
    -- Preferências
    preferences JSONB DEFAULT '{}'::jsonb,
    
    -- Timestamps
    last_login TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_plan CHECK (plan IN ('FREE_FOREVER', 'QUANT_STARTER', 'QUANT_PRO', 'QUANT_MASTER'))
);

-- Índices para users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_plan ON users(plan);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- ============================================
-- TABELA: user_profiles (dados adicionais)
-- ============================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Dados pessoais
    full_name VARCHAR(255),
    avatar_url TEXT,
    bio TEXT,
    
    -- Configurações de trading
    default_capital DECIMAL(15, 2) DEFAULT 10000.00,
    default_cdi DECIMAL(5, 4) DEFAULT 0.12,
    timezone VARCHAR(50) DEFAULT 'America/Sao_Paulo',
    
    -- Configurações de comissão
    default_broker_fee DECIMAL(10, 2) DEFAULT 0.50,
    default_exchange_fee DECIMAL(5, 4) DEFAULT 0.03,
    
    -- Notificações
    email_notifications BOOLEAN DEFAULT true,
    push_notifications BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id)
);

-- ============================================
-- TABELA: special_events (eventos especiais)
-- ============================================
CREATE TABLE IF NOT EXISTS special_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Dados do evento
    event_date DATE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    event_type VARCHAR(50) DEFAULT 'economic',
    impact VARCHAR(20) DEFAULT 'medium',
    
    -- Metadados
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_event_type CHECK (event_type IN ('economic', 'earnings', 'political', 'other')),
    CONSTRAINT valid_impact CHECK (impact IN ('high', 'medium', 'low'))
);

-- Índices para special_events
CREATE INDEX IF NOT EXISTS idx_events_user_id ON special_events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_date ON special_events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_type ON special_events(event_type);

-- ============================================
-- TABELA: saved_analyses (análises salvas)
-- ============================================
CREATE TABLE IF NOT EXISTS saved_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identificação
    title VARCHAR(255) NOT NULL,
    description TEXT,
    analysis_type VARCHAR(50) NOT NULL,
    
    -- Dados da análise
    file_name VARCHAR(255),
    file_size INTEGER,
    data JSONB NOT NULL,
    
    -- Metadados
    tags TEXT[],
    is_public BOOLEAN DEFAULT false,
    views_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_analysis_type CHECK (analysis_type IN ('backtest', 'daily', 'portfolio', 'correlation', 'discipline'))
);

-- Índices para saved_analyses
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON saved_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_type ON saved_analyses(analysis_type);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON saved_analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_tags ON saved_analyses USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_analyses_data ON saved_analyses USING GIN(data);

-- ============================================
-- TABELA: quantum_diary (diário quântico)
-- ============================================
CREATE TABLE IF NOT EXISTS quantum_diary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Data da entrada
    entry_date DATE NOT NULL,
    
    -- Dados de trading
    trades_data JSONB NOT NULL,
    performance_metrics JSONB,
    
    -- Análise emocional
    emotional_state VARCHAR(50),
    discipline_score DECIMAL(5, 2),
    notes TEXT,
    
    -- Objetivos e metas
    daily_goal DECIMAL(15, 2),
    achieved_goal BOOLEAN DEFAULT false,
    
    -- Metadados
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, entry_date)
);

-- Índices para quantum_diary
CREATE INDEX IF NOT EXISTS idx_diary_user_id ON quantum_diary(user_id);
CREATE INDEX IF NOT EXISTS idx_diary_entry_date ON quantum_diary(entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_diary_emotional_state ON quantum_diary(emotional_state);

-- ============================================
-- TABELA: portfolios (portfolio manager)
-- ============================================
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identificação
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Configurações
    initial_capital DECIMAL(15, 2) NOT NULL,
    current_capital DECIMAL(15, 2),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',
    
    -- Metadados
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'archived'))
);

-- Índices para portfolios
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_status ON portfolios(status);

-- ============================================
-- TABELA: portfolio_strategies (estratégias do portfolio)
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    
    -- Identificação da estratégia
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Alocação
    allocation_percentage DECIMAL(5, 2) NOT NULL,
    
    -- Performance
    performance_data JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_allocation CHECK (allocation_percentage >= 0 AND allocation_percentage <= 100)
);

-- Índices para portfolio_strategies
CREATE INDEX IF NOT EXISTS idx_strategies_portfolio_id ON portfolio_strategies(portfolio_id);

-- ============================================
-- TABELA: portfolio_trades (trades do portfolio)
-- ============================================
CREATE TABLE IF NOT EXISTS portfolio_trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES portfolio_strategies(id) ON DELETE SET NULL,
    
    -- Dados do trade
    entry_date TIMESTAMP WITH TIME ZONE NOT NULL,
    exit_date TIMESTAMP WITH TIME ZONE,
    symbol VARCHAR(50) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    
    -- Preços e quantidades
    entry_price DECIMAL(15, 4) NOT NULL,
    exit_price DECIMAL(15, 4),
    quantity INTEGER NOT NULL,
    
    -- Resultados
    pnl DECIMAL(15, 2),
    pnl_percentage DECIMAL(8, 4),
    
    -- Custos
    broker_fee DECIMAL(10, 2),
    exchange_fee DECIMAL(10, 2),
    
    -- Metadados
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_direction CHECK (direction IN ('long', 'short'))
);

-- Índices para portfolio_trades
CREATE INDEX IF NOT EXISTS idx_trades_portfolio_id ON portfolio_trades(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON portfolio_trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON portfolio_trades(entry_date DESC);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON portfolio_trades(symbol);

-- ============================================
-- TABELA: asset_costs (custos por ativo)
-- ============================================
CREATE TABLE IF NOT EXISTS asset_costs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Identificação do ativo
    symbol VARCHAR(50) NOT NULL,
    
    -- Custos
    broker_fee DECIMAL(10, 2) NOT NULL,
    exchange_fee DECIMAL(5, 4) NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, symbol)
);

-- Índices para asset_costs
CREATE INDEX IF NOT EXISTS idx_costs_user_id ON asset_costs(user_id);
CREATE INDEX IF NOT EXISTS idx_costs_symbol ON asset_costs(symbol);

-- ============================================
-- TABELA: api_keys (chaves de API para usuários)
-- ============================================
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Chave
    key_name VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    api_secret TEXT, -- Encrypted
    
    -- Permissões
    permissions TEXT[] DEFAULT ARRAY['read'],
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_permissions CHECK (
        permissions <@ ARRAY['read', 'write', 'delete', 'admin']::text[]
    )
);

-- Índices para api_keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(api_key);

-- ============================================
-- TABELA: usage_logs (logs de uso)
-- ============================================
CREATE TABLE IF NOT EXISTS usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Ação
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(50) NOT NULL,
    amount INTEGER DEFAULT 1,
    
    -- Metadados
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para usage_logs
CREATE INDEX IF NOT EXISTS idx_logs_user_id ON usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON usage_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_action ON usage_logs(action);

-- ============================================
-- VIEWS (visualizações úteis)
-- ============================================

-- View: Resumo de uso por usuário
CREATE OR REPLACE VIEW user_usage_summary AS
SELECT 
    u.id,
    u.email,
    u.name,
    u.plan,
    u.tokens_used,
    u.portfolios_created,
    u.analyses_run,
    COUNT(DISTINCT sa.id) AS total_analyses,
    COUNT(DISTINCT p.id) AS total_portfolios,
    COUNT(DISTINCT qd.id) AS diary_entries,
    u.last_login,
    u.created_at
FROM users u
LEFT JOIN saved_analyses sa ON sa.user_id = u.id
LEFT JOIN portfolios p ON p.user_id = u.id
LEFT JOIN quantum_diary qd ON qd.user_id = u.id
GROUP BY u.id;

-- View: Performance de portfolios
CREATE OR REPLACE VIEW portfolio_performance AS
SELECT 
    p.id AS portfolio_id,
    p.name,
    p.user_id,
    p.initial_capital,
    p.current_capital,
    COUNT(pt.id) AS total_trades,
    SUM(CASE WHEN pt.pnl > 0 THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN pt.pnl < 0 THEN 1 ELSE 0 END) AS losing_trades,
    COALESCE(SUM(pt.pnl), 0) AS total_pnl,
    ROUND(
        CASE 
            WHEN COUNT(pt.id) > 0 
            THEN (SUM(CASE WHEN pt.pnl > 0 THEN 1 ELSE 0 END)::DECIMAL / COUNT(pt.id) * 100)
            ELSE 0 
        END, 2
    ) AS win_rate,
    p.status,
    p.created_at,
    p.updated_at
FROM portfolios p
LEFT JOIN portfolio_trades pt ON pt.portfolio_id = p.id
GROUP BY p.id;

-- ============================================
-- TRIGGERS (atualizações automáticas)
-- ============================================

-- Trigger: Atualizar updated_at automaticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Aplicar trigger em todas as tabelas relevantes
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON user_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_special_events_updated_at BEFORE UPDATE ON special_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_saved_analyses_updated_at BEFORE UPDATE ON saved_analyses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_quantum_diary_updated_at BEFORE UPDATE ON quantum_diary
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolio_strategies_updated_at BEFORE UPDATE ON portfolio_strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_asset_costs_updated_at BEFORE UPDATE ON asset_costs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- FUNÇÕES AUXILIARES
-- ============================================

-- Função: Verificar limite de plano
CREATE OR REPLACE FUNCTION check_plan_limit(
    p_user_id UUID,
    p_resource VARCHAR(50),
    p_amount INTEGER DEFAULT 1
)
RETURNS BOOLEAN AS $$
DECLARE
    v_plan VARCHAR(50);
    v_current_usage INTEGER;
    v_limit INTEGER;
BEGIN
    -- Buscar plano e uso atual
    SELECT plan INTO v_plan FROM users WHERE id = p_user_id;
    
    -- Definir limites por plano
    CASE v_plan
        WHEN 'FREE_FOREVER' THEN
            v_limit := CASE p_resource
                WHEN 'tokens' THEN 100
                WHEN 'portfolios' THEN 0
                WHEN 'analyses' THEN 5
                ELSE 0
            END;
        WHEN 'QUANT_STARTER' THEN
            v_limit := CASE p_resource
                WHEN 'tokens' THEN 500
                WHEN 'portfolios' THEN 0
                WHEN 'analyses' THEN 20
                ELSE 0
            END;
        WHEN 'QUANT_PRO' THEN
            v_limit := CASE p_resource
                WHEN 'tokens' THEN 5000
                WHEN 'portfolios' THEN 5
                WHEN 'analyses' THEN -1  -- Ilimitado
                ELSE 0
            END;
        WHEN 'QUANT_MASTER' THEN
            v_limit := -1;  -- Tudo ilimitado
        ELSE
            v_limit := 0;
    END CASE;
    
    -- Se ilimitado, retornar true
    IF v_limit = -1 THEN
        RETURN TRUE;
    END IF;
    
    -- Verificar uso atual
    SELECT 
        CASE p_resource
            WHEN 'tokens' THEN tokens_used
            WHEN 'portfolios' THEN portfolios_created
            WHEN 'analyses' THEN analyses_run
            ELSE 0
        END
    INTO v_current_usage
    FROM users
    WHERE id = p_user_id;
    
    -- Verificar se há espaço
    RETURN (v_current_usage + p_amount) <= v_limit;
END;
$$ LANGUAGE plpgsql;

-- Função: Resetar uso mensal
CREATE OR REPLACE FUNCTION reset_monthly_usage()
RETURNS void AS $$
BEGIN
    UPDATE users SET
        tokens_used = 0,
        analyses_run = 0;
    
    INSERT INTO usage_logs (user_id, action, resource, amount)
    SELECT id, 'reset_monthly', 'all', 0
    FROM users;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- DADOS INICIAIS (SEED)
-- ============================================

-- Inserir usuário demo (opcional)
INSERT INTO users (email, name, plan, tokens_used, portfolios_created, analyses_run)
VALUES ('demo@devhubtrader.com', 'Usuário Demo', 'FREE_FOREVER', 0, 0, 0)
ON CONFLICT (email) DO NOTHING;

-- ============================================
-- PERMISSÕES
-- ============================================

-- Conceder permissões ao usuário do aplicativo
-- (ajustar conforme seu setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- ============================================
-- ÍNDICES DE PERFORMANCE
-- ============================================

-- Índice composto para queries comuns
CREATE INDEX IF NOT EXISTS idx_analyses_user_type_date 
    ON saved_analyses(user_id, analysis_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_diary_user_date 
    ON quantum_diary(user_id, entry_date DESC);

CREATE INDEX IF NOT EXISTS idx_trades_portfolio_date 
    ON portfolio_trades(portfolio_id, entry_date DESC);

-- ============================================
-- COMENTÁRIOS (DOCUMENTAÇÃO)
-- ============================================

COMMENT ON TABLE users IS 'Tabela principal de usuários do sistema';
COMMENT ON TABLE user_profiles IS 'Perfis e configurações adicionais dos usuários';
COMMENT ON TABLE special_events IS 'Eventos especiais para análise de impacto em trades';
COMMENT ON TABLE saved_analyses IS 'Análises de backtest salvas pelos usuários';
COMMENT ON TABLE quantum_diary IS 'Diário quântico com entradas diárias de trading';
COMMENT ON TABLE portfolios IS 'Portfolios gerenciados pelos usuários';
COMMENT ON TABLE portfolio_strategies IS 'Estratégias dentro de cada portfolio';
COMMENT ON TABLE portfolio_trades IS 'Trades executados em cada portfolio';
COMMENT ON TABLE asset_costs IS 'Custos personalizados por ativo e usuário';
COMMENT ON TABLE api_keys IS 'Chaves de API para integração externa';
COMMENT ON TABLE usage_logs IS 'Logs de uso de recursos do sistema';

-- ============================================
-- FIM DO SCHEMA
-- ============================================

