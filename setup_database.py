#!/usr/bin/env python
"""
Setup Database - DevHub Trader
Script para inicializar e configurar o banco de dados
"""

import os
import sys

def main():
    print("=" * 60)
    print("  DevHub Trader - Database Setup")
    print("=" * 60)
    print()
    
    # Verificar tipo de banco
    db_type = os.getenv('DB_TYPE', 'sqlite')
    
    print(f"📊 Tipo de banco: {db_type.upper()}")
    print()
    
    if db_type == 'postgresql':
        print("🔧 PostgreSQL detectado")
        print("   Execute o arquivo database_schema.sql no seu PostgreSQL:")
        print()
        print("   psql -U seu_usuario -d devhubtrader -f database_schema.sql")
        print()
        print("   Ou usando Docker:")
        print("   docker exec -i postgres_container psql -U postgres < database_schema.sql")
        print()
    else:
        print("🔧 SQLite detectado")
        print("   Inicializando banco de dados...")
        print()
        
        try:
            from database import db
            db.init_database()
            print("   ✅ Banco de dados criado com sucesso!")
            print(f"   📁 Arquivo: {db.db_url}")
            print()
        except Exception as e:
            print(f"   ❌ Erro ao criar banco: {e}")
            sys.exit(1)
    
    # Perguntar sobre migração
    print("=" * 60)
    print("  Migração de Dados")
    print("=" * 60)
    print()
    print("Deseja migrar dados em memória para o banco de dados?")
    print("(Isso irá copiar usuários, eventos, etc da memória para o DB)")
    print()
    
    resposta = input("Migrar dados? (s/n): ").lower().strip()
    
    if resposta == 's':
        print()
        print("🔄 Iniciando migração...")
        try:
            from database import migrate_memory_to_db
            migrate_memory_to_db()
            print("✅ Migração concluída com sucesso!")
        except Exception as e:
            print(f"❌ Erro na migração: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⏭️  Migração cancelada.")
    
    print()
    print("=" * 60)
    print("  Configuração do Ambiente")
    print("=" * 60)
    print()
    print("Para ATIVAR o banco de dados no sistema, configure:")
    print()
    print("   export USE_DATABASE=true")
    print()
    print("Ou crie um arquivo .env com:")
    print()
    print("   USE_DATABASE=true")
    print("   DB_TYPE=sqlite")
    print(f"   DATABASE_URL={os.path.abspath('devhubtrader.db') if db_type == 'sqlite' else 'postgresql://localhost/devhubtrader'}")
    print()
    print("=" * 60)
    print("  ✅ Setup Concluído!")
    print("=" * 60)
    print()

if __name__ == '__main__':
    main()

