#!/usr/bin/env python3
"""
Script para iniciar o backend da API DevHub Trader
"""

import os
import sys
import subprocess
import time
import signal
import psutil

def find_process_on_port(port):
    """Encontra processo rodando na porta especificada"""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.info['connections']:
                if conn.laddr.port == port:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def kill_process_on_port(port):
    """Mata processo rodando na porta especificada"""
    proc = find_process_on_port(port)
    if proc:
        print(f"🔄 Matando processo {proc.info['name']} (PID: {proc.info['pid']}) na porta {port}")
        proc.terminate()
        time.sleep(2)
        if proc.is_running():
            proc.kill()
        return True
    return False

def start_backend():
    """Inicia o backend"""
    port = 5002
    
    # Verifica se já há processo rodando na porta
    if find_process_on_port(port):
        print(f"⚠️  Já há um processo rodando na porta {port}")
        response = input("Deseja matar o processo existente e reiniciar? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operação cancelada")
            return
        
        kill_process_on_port(port)
    
    print("🚀 Iniciando backend na porta 5002...")
    print("📝 Logs do servidor:")
    print("=" * 50)
    
    try:
        # Inicia o servidor
        process = subprocess.Popen([
            sys.executable, "start_server.py"
        ], cwd=os.path.dirname(__file__))
        
        print(f"✅ Backend iniciado com PID: {process.pid}")
        print(f"🌐 API disponível em: http://localhost:{port}")
        print(f"📊 Endpoints disponíveis:")
        print(f"   - POST http://localhost:{port}/api/tabela")
        print(f"   - POST http://localhost:{port}/api/tabela-multipla")
        print(f"   - POST http://localhost:{port}/api/correlacao")
        print(f"   - POST http://localhost:{port}/api/disciplina-completa")
        print(f"   - POST http://localhost:{port}/api/trades")
        print(f"   - POST http://localhost:{port}/chat")
        print("=" * 50)
        print("💡 Pressione Ctrl+C para parar o servidor")
        
        # Aguarda o processo
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Parando servidor...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ Servidor parado")
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")

if __name__ == "__main__":
    start_backend() 