#!/bin/bash
# Script de deploy para Linux/Mac
# Deploy do backend e frontend para o servidor remoto

SERVER="root@31.97.241.157"
REMOTE_PATH_BACKEND="/root/python-freela/"
REMOTE_PATH_FRONTEND="/root/devhub-frontend/dist"
FRONTEND_PATH="$HOME/OneDrive/Área de Trabalho/desenvolvimento_lucas/devhubfront/devhubfront"

# Tentar caminho alternativo se o primeiro não existir
if [ ! -d "$FRONTEND_PATH" ]; then
    FRONTEND_PATH="/mnt/c/Users/lukas/OneDrive/Área de Trabalho/desenvolvimento_lucas/devhubfront/devhubfront"
fi

echo "🚀 Iniciando deploy para $SERVER"
echo "📁 Backend remoto: $REMOTE_PATH_BACKEND"
echo "📁 Frontend remoto: $REMOTE_PATH_FRONTEND"

# Copiar arquivos Python (excluindo __pycache__)
echo "📤 Copiando arquivos Python..."
scp *.py "$SERVER:$REMOTE_PATH_BACKEND"

# Copiar requirements.txt
echo "📤 Copiando requirements.txt..."
scp requirements.txt "$SERVER:$REMOTE_PATH_BACKEND"

# Copiar start_backend.sh se existir
if [ -f "start_backend.sh" ]; then
    echo "📤 Copiando start_backend.sh..."
    scp start_backend.sh "$SERVER:$REMOTE_PATH_BACKEND"
fi

echo "✅ Backend deployado com sucesso!"

# ============================================
# DEPLOY DO FRONTEND
# ============================================
echo ""
echo "========================================"
echo "Iniciando deploy do FRONTEND"
echo "========================================"

if [ ! -d "$FRONTEND_PATH" ]; then
    echo "❌ Caminho do frontend não encontrado: $FRONTEND_PATH"
    echo "⚠️  Continuando apenas com deploy do backend..."
else
    echo "🔍 Verificando build do frontend..."
    
    # Verificar se existe pasta dist no frontend
    if [ ! -d "$FRONTEND_PATH/dist" ]; then
        echo "📦 Pasta dist não encontrada. Executando build..."
        cd "$FRONTEND_PATH"
        npm run build
        if [ $? -ne 0 ]; then
            echo "❌ Erro ao fazer build do frontend!"
            echo "⚠️  Continuando apenas com deploy do backend..."
        else
            echo "✅ Build do frontend concluído!"
        fi
        cd - > /dev/null
    else
        echo "✅ Pasta dist encontrada!"
    fi
    
    # Verificar novamente se dist existe após build
    if [ -d "$FRONTEND_PATH/dist" ]; then
        echo "📤 Copiando arquivos do frontend para o servidor..."
        
        # Criar diretório remoto se não existir
        ssh $SERVER "mkdir -p $REMOTE_PATH_FRONTEND"
        
        # Copiar todos os arquivos da pasta dist recursivamente
        # Usar find para garantir que todos os arquivos sejam copiados
        file_count=0
        error_count=0
        
        find "$FRONTEND_PATH/dist" -type f | while read -r file; do
            # Calcular caminho relativo
            relative_path="${file#$FRONTEND_PATH/dist/}"
            remote_file="$REMOTE_PATH_FRONTEND/$relative_path"
            remote_dir=$(dirname "$remote_file")
            
            # Criar diretório remoto se necessário
            ssh $SERVER "mkdir -p \"$remote_dir\""
            
            # Copiar arquivo
            scp "$file" "$SERVER:$remote_file"
            
            if [ $? -eq 0 ]; then
                file_count=$((file_count + 1))
                if [ $((file_count % 10)) -eq 0 ]; then
                    echo "   ✅ $file_count arquivos copiados..."
                fi
            else
                error_count=$((error_count + 1))
                echo "   ❌ Erro ao copiar: $relative_path"
            fi
        done
        
        total_files=$(find "$FRONTEND_PATH/dist" -type f | wc -l)
        echo "✅ $total_files arquivos do frontend processados!"
    fi
fi

echo ""
echo "✅ Deploy completo concluído com sucesso!"
echo ""
echo "💡 Próximos passos no servidor:"
echo "   BACKEND:"
echo "   1. ssh $SERVER"
echo "   2. cd $REMOTE_PATH_BACKEND"
echo "   3. pip install -r requirements.txt"
echo "   4. python start_server.py"
echo ""
echo "   FRONTEND:"
echo "   Frontend deployado em: $REMOTE_PATH_FRONTEND"







