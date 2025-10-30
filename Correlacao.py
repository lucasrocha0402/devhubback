import pandas as pd
import numpy as np
from datetime import datetime

def calcular_correlacao_por_data_e_direcao(arquivos_df_list):
    """
    Função para analisar correlação por data, direção e resultado entre estratégias.
    
    ATENDE OS REQUISITOS:
    1. Operações na mesma data e mesma direção (compra/venda)
    2. Correlação por data: quando uma perde, a outra também perde?
    
    Parâmetros:
    arquivos_df_list: Lista de dicionários {'nome': 'arquivo', 'df': dataframe}
    """
    
    if len(arquivos_df_list) < 2:
        return {"erro": "Precisa de pelo menos 2 arquivos"}
    
    # Processar cada arquivo
    dados_processados = {}
    
    for arquivo in enumerate(arquivos_df_list):
        df = arquivo[1]['df'].copy()
        nome = arquivo[1]['nome']
        
        # Extrair data (sem horário)
        if 'entry_date' in df.columns:
            df['data'] = pd.to_datetime(df['entry_date']).dt.date
        elif 'Abertura' in df.columns:
            df['data'] = pd.to_datetime(df['Abertura']).dt.date
        else:
            df['data'] = pd.to_datetime(df.index).dt.date
        
        # Determinar direção da operação
        if 'Tipo' in df.columns:
            df['direcao'] = df['Tipo'].apply(lambda x: 'COMPRA' if 'C' in str(x).upper() else 'VENDA')
        elif 'Quantidade' in df.columns:
            df['direcao'] = df['Quantidade'].apply(lambda x: 'COMPRA' if x > 0 else 'VENDA')
        else:
            # Assumir todas como COMPRA se não conseguir determinar
            df['direcao'] = 'COMPRA'
        
        # Determinar coluna de resultado
        if 'operation_result' in df.columns:
            result_col = 'operation_result'
        elif 'pnl' in df.columns:
            result_col = 'pnl'
        elif 'Res. Operação' in df.columns:
            result_col = 'Res. Operação'
        else:
            result_col = None
        
        if result_col is None:
            continue
        
        # Agrupar por data e direção
        agrupado = df.groupby(['data', 'direcao']).agg({
            result_col: ['sum', 'count', 'mean']
        }).round(2)
        
        # Flatten colunas
        agrupado.columns = ['resultado_total', 'num_operacoes', 'resultado_medio']
        agrupado = agrupado.reset_index()
        
        # Também agrupar só por data (total do dia)
        resultado_dia = df.groupby('data')[result_col].agg([
            'sum', 'count'
        ]).round(2)
        resultado_dia.columns = ['resultado_dia_total', 'operacoes_dia_total']
        
        dados_processados[nome] = {
            'por_data_direcao': agrupado,
            'por_data': resultado_dia
        }
    
    # ANÁLISE 1: CORRELAÇÃO POR DATA E DIREÇÃO
    # Encontrar datas e direções em comum
    correlacao_direcao = analisar_correlacao_direcao(dados_processados, arquivos_df_list)
    
    # ANÁLISE 2: CORRELAÇÃO POR DATA E RESULTADO  
    # "Quando uma perde, a outra também perde?"
    correlacao_resultado = analisar_correlacao_resultado(dados_processados, arquivos_df_list)
    
    return {
        "correlacao_data_direcao": correlacao_direcao,
        "correlacao_data_resultado": correlacao_resultado
    }

def analisar_correlacao_direcao(dados_processados, arquivos_df_list):
    """Analisa operações na mesma data e mesma direção"""
    
    nomes = [arq['nome'] for arq in arquivos_df_list]
    
    # Encontrar todas as combinações data+direção
    todas_combinacoes = set()
    for nome in nomes:
        df_direcao = dados_processados[nome]['por_data_direcao']
        for _, row in df_direcao.iterrows():
            todas_combinacoes.add((row['data'], row['direcao']))
    
    # Analisar cada combinação
    analise_direcao = []
    
    for data, direcao in sorted(todas_combinacoes):
        operacao = {
            "data": data.strftime('%Y-%m-%d'),
            "direcao": direcao,
            "estrategias": {},
            "analise": {}
        }
        
        # Ver qual estratégia operou nesta data+direção
        estrategias_operaram = []
        resultados = []
        
        for nome in nomes:
            df_direcao = dados_processados[nome]['por_data_direcao']
            
            # Filtrar por data e direção
            operacoes = df_direcao[
                (df_direcao['data'] == data) & 
                (df_direcao['direcao'] == direcao)
            ]
            
            if not operacoes.empty:
                resultado = operacoes['resultado_total'].iloc[0]
                num_ops = operacoes['num_operacoes'].iloc[0]
                
                operacao["estrategias"][nome] = {
                    "operou": True,
                    "resultado": float(resultado),
                    "num_operacoes": int(num_ops),
                    "status": "ganhou" if resultado > 0 else "perdeu" if resultado < 0 else "empate"
                }
                
                estrategias_operaram.append(nome)
                resultados.append(resultado)
            else:
                operacao["estrategias"][nome] = {
                    "operou": False,
                    "resultado": 0,
                    "num_operacoes": 0,
                    "status": "nao_operou"
                }
        
        # Análise da correlação nesta data+direção
        if len(estrategias_operaram) >= 2:
            positivos = sum(1 for r in resultados if r > 0)
            negativos = sum(1 for r in resultados if r < 0)
            
            operacao["analise"] = {
                "multiplas_estrategias": True,
                "todas_ganharam": positivos == len(resultados),
                "todas_perderam": negativos == len(resultados),
                "resultados_mistos": positivos > 0 and negativos > 0,
                "resultado_combinado": float(sum(resultados))
            }
        else:
            operacao["analise"] = {
                "multiplas_estrategias": False,
                "apenas_uma_operou": len(estrategias_operaram) == 1
            }
        
        analise_direcao.append(operacao)
    
    # Estatísticas resumo
    ops_multiplas = [op for op in analise_direcao if op["analise"].get("multiplas_estrategias", False)]
    total_ops_multiplas = len(ops_multiplas)
    
    if total_ops_multiplas > 0:
        todas_ganharam = sum(1 for op in ops_multiplas if op["analise"]["todas_ganharam"])
        todas_perderam = sum(1 for op in ops_multiplas if op["analise"]["todas_perderam"])
        mistas = sum(1 for op in ops_multiplas if op["analise"]["resultados_mistos"])
        
        resumo_direcao = {
            "total_operacoes_simultaneas": total_ops_multiplas,
            "todas_ganharam_simultaneo": todas_ganharam,
            "todas_perderam_simultaneo": todas_perderam, 
            "resultados_mistos": mistas,
            "pct_correlacao_positiva": round((todas_ganharam / total_ops_multiplas) * 100, 1) if total_ops_multiplas > 0 else 0,
            "pct_correlacao_negativa": round((todas_perderam / total_ops_multiplas) * 100, 1) if total_ops_multiplas > 0 else 0,
            "pct_diversificacao": round((mistas / total_ops_multiplas) * 100, 1) if total_ops_multiplas > 0 else 0
        }
    else:
        resumo_direcao = {"erro": "Não há operações simultâneas na mesma direção"}
    
    return {
        "resumo": resumo_direcao,
        "detalhes": analise_direcao[:20]  # Primeiros 20 para não sobrecarregar
    }

def analisar_correlacao_resultado(dados_processados, arquivos_df_list):
    """Analisa: quando uma perde, a outra também perde no mesmo dia?"""
    
    nomes = [arq['nome'] for arq in arquivos_df_list]
    
    # Encontrar datas em comum
    datas_comuns = set(dados_processados[nomes[0]]['por_data'].index)
    for nome in nomes[1:]:
        datas_comuns = datas_comuns.intersection(set(dados_processados[nome]['por_data'].index))
    
    datas_comuns = sorted(list(datas_comuns))
    
    if not datas_comuns:
        return {"erro": "Não há datas em comum"}
    
    # Analisar cada data
    analise_resultado = []
    
    for data in datas_comuns:
        dia = {
            "data": data.strftime('%Y-%m-%d'),
            "estrategias": {},
            "analise": {}
        }
        
        resultados_dia = []
        
        for nome in nomes:
            resultado_dia = dados_processados[nome]['por_data'].loc[data, 'resultado_dia_total']
            num_ops = dados_processados[nome]['por_data'].loc[data, 'operacoes_dia_total']
            
            dia["estrategias"][nome] = {
                "resultado_dia": float(resultado_dia),
                "num_operacoes": int(num_ops),
                "status": "ganhou" if resultado_dia > 0 else "perdeu" if resultado_dia < 0 else "empate"
            }
            
            resultados_dia.append(resultado_dia)
        
        # Análise do dia
        positivos = sum(1 for r in resultados_dia if r > 0)
        negativos = sum(1 for r in resultados_dia if r < 0)
        zeros = sum(1 for r in resultados_dia if r == 0)
        
        dia["analise"] = {
            "todas_ganharam": positivos == len(resultados_dia),
            "todas_perderam": negativos == len(resultados_dia),
            "resultados_mistos": positivos > 0 and negativos > 0,
            "resultado_combinado": float(sum(resultados_dia)),
            "correlacao_direcional": (positivos == len(resultados_dia)) or (negativos == len(resultados_dia))
        }
        
        analise_resultado.append(dia)
    
    # Estatísticas resumo - RESPONDE A PERGUNTA PRINCIPAL
    total_dias = len(analise_resultado)
    dias_todas_ganharam = sum(1 for dia in analise_resultado if dia["analise"]["todas_ganharam"])
    dias_todas_perderam = sum(1 for dia in analise_resultado if dia["analise"]["todas_perderam"])
    dias_mistos = sum(1 for dia in analise_resultado if dia["analise"]["resultados_mistos"])
    
    resumo_resultado = {
        "total_dias_analisados": total_dias,
        "dias_todas_ganharam": dias_todas_ganharam,
        "dias_todas_perderam": dias_todas_perderam,
        "dias_resultados_mistos": dias_mistos,
        "resposta_pergunta": {
            "quando_uma_perde_outra_tambem_perde_pct": round((dias_todas_perderam / total_dias) * 100, 1) if total_dias > 0 else 0,
            "quando_uma_ganha_outra_tambem_ganha_pct": round((dias_todas_ganharam / total_dias) * 100, 1) if total_dias > 0 else 0,
            "dias_com_diversificacao_pct": round((dias_mistos / total_dias) * 100, 1) if total_dias > 0 else 0
        },
        "interpretacao": (
            "Alta correlação" if total_dias > 0 and (dias_todas_ganharam + dias_todas_perderam) / total_dias > 0.7 
            else "Boa diversificação" if total_dias > 0 and dias_mistos / total_dias > 0.5 
            else "Correlação moderada"
        )
    }
    
    return {
        "resumo": resumo_resultado,
        "detalhes": analise_resultado[:30]  # Primeiros 30 dias
    }