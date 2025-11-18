import pandas as pd
from flask import jsonify

def processar_multiplos_arquivos(files, carregar_csv_func, calcular_performance_func, calcular_day_of_week_func, calcular_monthly_func):
    """
    Processa múltiplos arquivos CSV e retorna análise consolidada
    
    Args:
        files: Lista de arquivos do request.files.getlist('files')
        carregar_csv_func: Função para carregar CSV
        calcular_performance_func: Função para calcular performance
        calcular_day_of_week_func: Função para análise por dia da semana
        calcular_monthly_func: Função para análise mensal
    
    Returns:
        dict: Resultado consolidado ou dict com erro
    """
    try:
        # ✅ CORREÇÃO: Verifica se tem arquivos com otimizações
        if not files or all(f.filename == '' for f in files):
            return {"error": "Nenhum arquivo válido enviado"}, 400
        
        # ✅ CORREÇÃO: Processa todos os arquivos e consolida com otimizações
        dataframes = []
        arquivos_processados = []
        
        for file in files:
            if file.filename == '':
                continue
                
            try:
                df = carregar_csv_func(file)
                dataframes.append(df)
                arquivos_processados.append({
                    'nome': file.filename,
                    'trades': len(df)
                })
            except Exception as e:
                return {"error": f"Erro no arquivo {file.filename}: {str(e)}"}, 400
        
        if not dataframes:
            return {"error": "Nenhum arquivo foi processado com sucesso"}, 400
        
        # ✅ CORREÇÃO: Consolida todos os DataFrames com otimizações
        df_consolidado = pd.concat(dataframes, ignore_index=True)
        df_consolidado = df_consolidado.sort_values('Abertura')
        
        # ✅ CORREÇÃO: Calcula as métricas consolidadas com otimizações
        performance = calcular_performance_func(df_consolidado)
        dow = calcular_day_of_week_func(df_consolidado)
        monthly = calcular_monthly_func(df_consolidado)
        
        # ✅ CORREÇÃO: Retorna resultado igual ao formato original, mas com info dos arquivos
        resultado = {
            "arquivos_info": {
                "total_arquivos": len(arquivos_processados),
                "arquivos": arquivos_processados,
                "total_trades_consolidado": len(df_consolidado)
            },
            "Performance Metrics": performance,
            "Day of Week Analysis": dow,
            "Monthly Analysis": monthly
        }

        return resultado, 200

    except Exception as e:
        return {"error": str(e)}, 500


def processar_multiplos_arquivos_comparativo(files, carregar_csv_func, calcular_performance_func):
    """
    Processa múltiplos arquivos CSV e retorna comparativo entre eles
    
    Args:
        files: Lista de arquivos do request.files.getlist('files')
        carregar_csv_func: Função para carregar CSV
        calcular_performance_func: Função para calcular performance
    
    Returns:
        dict: Comparativo entre arquivos ou dict com erro
    """
    try:
        # ✅ CORREÇÃO: Verifica se tem arquivos com otimizações
        if not files or all(f.filename == '' for f in files):
            return {"error": "Nenhum arquivo válido enviado"}, 400
        
        if len(files) < 2:
            return {"error": "É necessário pelo menos 2 arquivos para comparação"}, 400
        
        # ✅ CORREÇÃO: Processa cada arquivo individualmente com otimizações
        comparativo = {}
        
        for file in files:
            if file.filename == '':
                continue
                
            try:
                df = carregar_csv_func(file)
                performance = calcular_performance_func(df)
                
                comparativo[file.filename] = {
                    'total_trades': len(df),
                    'net_profit': performance['Net Profit'],
                    'profit_factor': performance['Profit Factor'],
                    'win_rate': performance['Win Rate (%)'],
                    'max_drawdown': performance['Max Drawdown ($)'],
                    'sharpe_ratio': performance['Sharpe Ratio'],
                    'average_per_trade': performance['Average PnL/Trade'],
                    'recovery_factor': performance['Recovery Factor']
                }
                
            except Exception as e:
                return {"error": f"Erro no arquivo {file.filename}: {str(e)}"}, 400
        
        if not comparativo:
            return {"error": "Nenhum arquivo foi processado com sucesso"}, 400
        
        # ✅ CORREÇÃO: Adiciona ranking com otimizações
        resultado = {
            "comparativo": comparativo,
            "ranking": {
                "melhor_profit": max(comparativo.items(), key=lambda x: x[1]['net_profit'])[0],
                "melhor_profit_factor": max(comparativo.items(), key=lambda x: x[1]['profit_factor'])[0],
                "melhor_win_rate": max(comparativo.items(), key=lambda x: x[1]['win_rate'])[0],
                "menor_drawdown": min(comparativo.items(), key=lambda x: abs(x[1]['max_drawdown']))[0],
                "melhor_sharpe": max(comparativo.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
            }
        }

        return resultado, 200

    except Exception as e:
        return {"error": str(e)}, 500
