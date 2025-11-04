#!/usr/bin/env python
"""
Test Database - DevHub Trader
Script para testar todas as funcionalidades do banco de dados
"""

import os
import sys
from datetime import datetime

# Configurar para usar SQLite em modo teste
os.environ['USE_DATABASE'] = 'true'
os.environ['DB_TYPE'] = 'sqlite'
os.environ['DATABASE_URL'] = 'test_devhubtrader.db'

def test_database_connection():
    """Teste 1: Conexão com banco de dados"""
    print("=" * 60)
    print("TESTE 1: Conexão com Banco de Dados")
    print("=" * 60)
    
    try:
        from database import db
        db.init_database()
        print("✅ Banco de dados inicializado com sucesso!")
        return True
    except Exception as e:
        print(f"❌ Erro ao inicializar banco: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_user_manager():
    """Teste 2: Gerenciamento de usuários"""
    print("\n" + "=" * 60)
    print("TESTE 2: Gerenciamento de Usuários")
    print("=" * 60)
    
    try:
        from database import user_manager
        
        # Criar usuário
        user_id = user_manager.create_user(
            email='teste@devhubtrader.com',
            name='Usuário Teste',
            plan='QUANT_PRO'
        )
        print(f"✅ Usuário criado: {user_id}")
        
        # Buscar usuário
        user = user_manager.get_user(user_id=user_id)
        print(f"✅ Usuário encontrado: {user.get('name')} ({user.get('email')})")
        
        # Atualizar usuário
        user_manager.update_user(user_id, tokens_used=50)
        print(f"✅ Usuário atualizado")
        
        # Verificar limite
        check = user_manager.check_limit(user_id, 'tokens', 100)
        print(f"✅ Verificação de limite: {check}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste de usuários: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_event_manager():
    """Teste 3: Eventos especiais"""
    print("\n" + "=" * 60)
    print("TESTE 3: Eventos Especiais")
    print("=" * 60)
    
    try:
        from database import event_manager
        
        # Criar evento
        event_id = event_manager.create_event(
            user_id='admin',
            date='2024-01-15',
            name='FOMC Meeting',
            description='Federal Reserve meeting',
            event_type='economic',
            impact='high'
        )
        print(f"✅ Evento criado: {event_id}")
        
        # Listar eventos
        events = event_manager.get_events()
        print(f"✅ Total de eventos: {len(events)}")
        
        for event in events:
            print(f"   📅 {event.get('event_date')}: {event.get('name')}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste de eventos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_manager():
    """Teste 4: Análises salvas"""
    print("\n" + "=" * 60)
    print("TESTE 4: Análises Salvas")
    print("=" * 60)
    
    try:
        from database import analysis_manager
        
        # Salvar análise
        analysis_id = analysis_manager.save_analysis(
            user_id='teste_user',
            title='Backtest WDO Janeiro 2024',
            analysis_type='backtest',
            data={
                'total_trades': 150,
                'win_rate': 58.3,
                'profit_factor': 1.85,
                'total_pnl': 5420.50
            },
            description='Análise de teste',
            file_name='wdo_jan_2024.csv'
        )
        print(f"✅ Análise salva: {analysis_id}")
        
        # Listar análises
        analyses = analysis_manager.get_analyses('teste_user')
        print(f"✅ Total de análises: {len(analyses)}")
        
        for analysis in analyses:
            print(f"   📊 {analysis.get('title')}: {analysis.get('data', {}).get('total_trades', 0)} trades")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste de análises: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diary_manager():
    """Teste 5: Diário Quântico"""
    print("\n" + "=" * 60)
    print("TESTE 5: Diário Quântico")
    print("=" * 60)
    
    try:
        from database import diary_manager
        
        # Salvar entrada
        entry_id = diary_manager.save_entry(
            user_id='teste_user',
            entry_date='2024-01-15',
            trades_data={
                'trades': 5,
                'pnl': 250.50,
                'win_rate': 60.0
            },
            performance_metrics={
                'profit_factor': 1.8,
                'drawdown': -45.20
            },
            emotional_state='disciplinado',
            notes='Dia produtivo, segui o plano à risca.'
        )
        print(f"✅ Entrada do diário salva: {entry_id}")
        
        # Buscar entrada
        entry = diary_manager.get_entry('teste_user', '2024-01-15')
        if entry:
            print(f"✅ Entrada encontrada: {entry.get('emotional_state')}")
            print(f"   Trades: {entry.get('trades_data', {}).get('trades')}")
            print(f"   PnL: R$ {entry.get('trades_data', {}).get('pnl')}")
        
        # Listar entradas
        entries = diary_manager.get_entries('teste_user')
        print(f"✅ Total de entradas: {len(entries)}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste do diário: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_manager():
    """Teste 6: Portfolio Manager"""
    print("\n" + "=" * 60)
    print("TESTE 6: Portfolio Manager")
    print("=" * 60)
    
    try:
        from database import portfolio_manager
        
        # Criar portfolio
        portfolio_id = portfolio_manager.create_portfolio(
            user_id='teste_user',
            name='Portfolio Agressivo',
            initial_capital=100000.00,
            description='Portfolio focado em crescimento'
        )
        print(f"✅ Portfolio criado: {portfolio_id}")
        
        # Listar portfolios
        portfolios = portfolio_manager.get_portfolios('teste_user')
        print(f"✅ Total de portfolios: {len(portfolios)}")
        
        for portfolio in portfolios:
            print(f"   💼 {portfolio.get('name')}: R$ {portfolio.get('initial_capital')}")
        
        # Atualizar portfolio
        portfolio_manager.update_portfolio(
            portfolio_id,
            current_capital=105000.00
        )
        print(f"✅ Portfolio atualizado")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste de portfolios: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_services():
    """Teste 7: Services (Wrappers)"""
    print("\n" + "=" * 60)
    print("TESTE 7: Services (db_integration.py)")
    print("=" * 60)
    
    try:
        from db_integration import UserService, EventService, AnalysisService, DiaryService
        
        # Testar UserService
        plan = UserService.get_user_plan('teste_user')
        print(f"✅ UserService.get_user_plan: {plan}")
        
        usage = UserService.get_user_usage('teste_user')
        print(f"✅ UserService.get_user_usage: {usage}")
        
        # Testar EventService
        events = EventService.list_events()
        print(f"✅ EventService.list_events: {len(events)} eventos")
        
        # Testar AnalysisService
        analyses = AnalysisService.get_analyses('teste_user')
        print(f"✅ AnalysisService.get_analyses: {len(analyses)} análises")
        
        # Testar DiaryService
        entries = DiaryService.get_entries('teste_user')
        print(f"✅ DiaryService.get_entries: {len(entries)} entradas")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste de services: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Limpar banco de teste"""
    print("\n" + "=" * 60)
    print("LIMPEZA")
    print("=" * 60)
    
    try:
        db_file = 'test_devhubtrader.db'
        if os.path.exists(db_file):
            os.remove(db_file)
            print(f"✅ Banco de teste removido: {db_file}")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao limpar: {e}")
        return False

def main():
    """Executar todos os testes"""
    print("\n" + "=" * 60)
    print("  DevHub Trader - Testes de Banco de Dados")
    print("=" * 60)
    print()
    
    tests = [
        ("Conexão", test_database_connection),
        ("Usuários", test_user_manager),
        ("Eventos", test_event_manager),
        ("Análises", test_analysis_manager),
        ("Diário Quântico", test_diary_manager),
        ("Portfolios", test_portfolio_manager),
        ("Services", test_services)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro crítico no teste '{name}': {e}")
            results.append((name, False))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("  RESUMO DOS TESTES")
    print("=" * 60)
    print()
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    failed = total - passed
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {name:.<30} {status}")
    
    print()
    print(f"  Total: {passed}/{total} testes passaram")
    print()
    
    if failed == 0:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso!")
    else:
        print(f"⚠️ {failed} teste(s) falharam. Verifique os erros acima.")
    
    print("=" * 60)
    print()
    
    # Perguntar sobre limpeza
    resposta = input("Remover banco de teste? (s/n): ").lower().strip()
    if resposta == 's':
        cleanup()
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

