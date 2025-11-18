#!/bin/bash

echo "========================================"
echo "    DevHub Trader Backend API"
echo "========================================"
echo ""
echo "Iniciando backend na porta 5002..."
echo ""
echo "Endpoints disponíveis:"
echo "  - POST http://localhost:5002/api/tabela"
echo "  - POST http://localhost:5002/api/tabela-multipla"
echo "  - POST http://localhost:5002/api/correlacao"
echo "  - POST http://localhost:5002/api/disciplina-completa"
echo "  - POST http://localhost:5002/api/trades"
echo "  - POST http://localhost:5002/chat"
echo ""
echo "Pressione Ctrl+C para parar o servidor"
echo "========================================"
echo ""

python start_server.py 