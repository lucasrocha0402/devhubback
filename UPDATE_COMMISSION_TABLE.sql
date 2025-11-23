-- Script para atualizar a tabela user_commission_settings
-- Execute este script no Supabase SQL Editor se a tabela já existir

-- Adicionar colunas para corretagem e emolumentos (se não existirem)
ALTER TABLE user_commission_settings 
ADD COLUMN IF NOT EXISTS corretagem JSONB DEFAULT '{"method": "fixed_per_roda", "value": 0.50, "override_existing": true}'::jsonb,
ADD COLUMN IF NOT EXISTS emolumentos JSONB DEFAULT '{"method": "fixed_per_roda", "value": 0.03, "override_existing": true}'::jsonb;

-- Migrar dados antigos para o novo formato (se houver dados antigos)
UPDATE user_commission_settings
SET 
  corretagem = jsonb_build_object(
    'method', CASE 
      WHEN default_method = 'fixed' THEN 'fixed_per_roda'
      WHEN default_method = 'percentage' THEN 'percentage'
      ELSE 'fixed_per_roda'
    END,
    'value', COALESCE(default_value * 0.94, 0.50),
    'override_existing', COALESCE(override_existing, true)
  ),
  emolumentos = jsonb_build_object(
    'method', 'fixed_per_roda',
    'value', COALESCE(default_value * 0.06, 0.03),
    'override_existing', COALESCE(override_existing, true)
  )
WHERE corretagem IS NULL OR emolumentos IS NULL;

-- Criar tabela do zero se não existir (versão completa)
CREATE TABLE IF NOT EXISTS user_commission_settings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Configurações de corretagem (novo formato)
  corretagem JSONB DEFAULT '{"method": "fixed_per_roda", "value": 0.50, "override_existing": true}'::jsonb,
  
  -- Configurações de emolumentos (novo formato)
  emolumentos JSONB DEFAULT '{"method": "fixed_per_roda", "value": 0.03, "override_existing": true}'::jsonb,
  
  -- Configurações gerais
  apply_difference_to_pnl BOOLEAN DEFAULT true,
  
  -- Configurações por ativo (armazenadas como JSON)
  asset_configs JSONB DEFAULT '[]'::jsonb,
  
  -- Campos legados (mantidos para compatibilidade)
  default_method TEXT,
  default_value DECIMAL(10, 4),
  override_existing BOOLEAN DEFAULT true,
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  UNIQUE(user_id)
);

-- Índice para buscar por user_id
CREATE INDEX IF NOT EXISTS idx_user_commission_settings_user_id ON user_commission_settings(user_id);

-- Trigger para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_user_commission_settings_updated_at ON user_commission_settings;
CREATE TRIGGER update_user_commission_settings_updated_at
  BEFORE UPDATE ON user_commission_settings
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- RLS (Row Level Security) - garantir que está habilitado
ALTER TABLE user_commission_settings ENABLE ROW LEVEL SECURITY;

-- Políticas RLS (recriar se necessário)
DROP POLICY IF EXISTS "Users can view their own commission settings" ON user_commission_settings;
CREATE POLICY "Users can view their own commission settings"
  ON user_commission_settings FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert their own commission settings" ON user_commission_settings;
CREATE POLICY "Users can insert their own commission settings"
  ON user_commission_settings FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can update their own commission settings" ON user_commission_settings;
CREATE POLICY "Users can update their own commission settings"
  ON user_commission_settings FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can delete their own commission settings" ON user_commission_settings;
CREATE POLICY "Users can delete their own commission settings"
  ON user_commission_settings FOR DELETE
  USING (auth.uid() = user_id);

