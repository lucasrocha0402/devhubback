#!/bin/bash
# Script de deploy para Linux/Mac
# Deploy do backend para o servidor remoto

SERVER="root@31.97.241.157"
REMOTE_PATH="/root/python-freela/"

echo "🚀 Iniciando deploy para $SERVER"
echo "📁 Diretório remoto: $REMOTE_PATH"

# Copiar arquivos Python (excluindo __pycache__)
echo "📤 Copiando arquivos Python..."
scp *.py "$SERVER:$REMOTE_PATH"

# Copiar requirements.txt
echo "📤 Copiando requirements.txt..."
scp requirements.txt "$SERVER:$REMOTE_PATH"

# Copiar start_backend.sh se existir
if [ -f "start_backend.sh" ]; then
    echo "📤 Copiando start_backend.sh..."
    scp start_backend.sh "$SERVER:$REMOTE_PATH"
fi

echo "✅ Deploy concluído com sucesso!"
echo ""
echo "💡 Próximos passos no servidor:"
echo "   1. ssh $SERVER"
echo "   2. cd $REMOTE_PATH"
echo "   3. pip install -r requirements.txt"
echo "   4. python start_server.py"

