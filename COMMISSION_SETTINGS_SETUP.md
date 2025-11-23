# Configuração do Sistema de Comissões - Backend

## Variáveis de Ambiente Necessárias

Adicione as seguintes variáveis ao seu arquivo `.env`:

```env
# Supabase Configuration
SUPABASE_URL=https://seu-projeto.supabase.co
SUPABASE_SERVICE_ROLE_KEY=sua-service-role-key
# OU
SUPABASE_ANON_KEY=sua-anon-key

# Opcional (recomendado para produção)
SUPABASE_JWT_SECRET=seu-jwt-secret
```

## Instalação das Dependências

```bash
pip install -r requirements.txt
```

As novas dependências adicionadas são:
- `supabase==2.3.4` - Cliente Python para Supabase
- `PyJWT==2.8.0` - Para validação de tokens JWT
- `cryptography==42.0.5` - Dependência do PyJWT

## Setup do Banco de Dados (Supabase)

Execute o seguinte SQL no Supabase SQL Editor:

```sql
CREATE TABLE user_commission_settings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Configurações padrão
  default_method TEXT NOT NULL DEFAULT 'fixed', -- 'fixed' ou 'percentage'
  default_value DECIMAL(10, 4) NOT NULL DEFAULT 0,
  override_existing BOOLEAN DEFAULT true,
  apply_difference_to_pnl BOOLEAN DEFAULT true,
  
  -- Configurações por ativo (armazenadas como JSON)
  asset_configs JSONB DEFAULT '[]'::jsonb, -- Array de objetos: [{asset: string, method: string, value: number}]
  
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  UNIQUE(user_id)
);

-- Índice para buscar por user_id
CREATE INDEX idx_user_commission_settings_user_id ON user_commission_settings(user_id);

-- Trigger para atualizar updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_commission_settings_updated_at
  BEFORE UPDATE ON user_commission_settings
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- RLS (Row Level Security)
ALTER TABLE user_commission_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own commission settings"
  ON user_commission_settings FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own commission settings"
  ON user_commission_settings FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own commission settings"
  ON user_commission_settings FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own commission settings"
  ON user_commission_settings FOR DELETE
  USING (auth.uid() = user_id);
```

## Endpoints Implementados

### GET `/api/user/commission-settings`

**Autenticação**: Requer header `Authorization: Bearer <token>`

**Resposta de Sucesso (200)**:
```json
{
  "defaultMethod": "fixed",
  "defaultValue": 0.25,
  "overrideExisting": true,
  "applyDifferenceToPnl": true,
  "configs": [
    {
      "asset": "WINFUT",
      "method": "fixed",
      "value": 0.25
    }
  ]
}
```

**Resposta quando não existe (200 com defaults)**:
```json
{
  "defaultMethod": "fixed",
  "defaultValue": 0,
  "overrideExisting": true,
  "applyDifferenceToPnl": true,
  "configs": []
}
```

### PUT `/api/user/commission-settings`

**Autenticação**: Requer header `Authorization: Bearer <token>`

**Request Body**:
```json
{
  "defaultMethod": "fixed",
  "defaultValue": 0.25,
  "overrideExisting": true,
  "applyDifferenceToPnl": true,
  "configs": [
    {
      "asset": "WINFUT",
      "method": "fixed",
      "value": 0.25
    }
  ]
}
```

**Validações**:
- `defaultMethod` deve ser "fixed" ou "percentage"
- `defaultValue` deve ser >= 0
- `configs` deve ser um array
- Cada item em `configs` deve ter:
  - `asset`: string não vazia
  - `method`: "fixed" ou "percentage"
  - `value`: número >= 0

**Resposta de Sucesso (200)**:
```json
{
  "success": true,
  "message": "Configurações salvas com sucesso"
}
```

## Segurança

- Os endpoints requerem autenticação via token JWT do Supabase
- O RLS (Row Level Security) no Supabase garante que usuários só acessem suas próprias configurações
- Recomenda-se usar `SUPABASE_JWT_SECRET` em produção para validação completa dos tokens

## Testando os Endpoints

### Exemplo com curl:

```bash
# GET
curl -X GET http://localhost:5002/api/user/commission-settings \
  -H "Authorization: Bearer SEU_TOKEN_JWT"

# PUT
curl -X PUT http://localhost:5002/api/user/commission-settings \
  -H "Authorization: Bearer SEU_TOKEN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "defaultMethod": "fixed",
    "defaultValue": 0.25,
    "overrideExisting": true,
    "applyDifferenceToPnl": true,
    "configs": [
      {
        "asset": "WINFUT",
        "method": "fixed",
        "value": 0.25
      }
    ]
  }'
```

## Próximos Passos no Frontend

1. Atualizar `commissionStore.ts` para adicionar:
   - `loadFromBackend()`: Chamar GET `/api/user/commission-settings`
   - `saveToBackend()`: Chamar PUT `/api/user/commission-settings`

2. Sincronização:
   - Carregar do backend ao fazer login
   - Salvar no backend quando houver mudanças (com debounce)
   - Manter localStorage como fallback


