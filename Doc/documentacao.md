# Documentação Simplificada da API

**Endereço base:**  
http://127.0.0.1:5000


## Iniciar

Para iniciar o servidor basta executar start_server.py

---
## ROTAS

## 1) Enviar CSV de trades  
**Rota:** `POST /api/tabela`

### O que faz  
Você manda o seu arquivo CSV com o histórico de operações e recebe de volta as principais métricas:

- Lucro líquido, bruto e perdas  
- Fator de lucro e de recuperação  
- Taxa de acerto (win rate) e médias de ganho/perda  
- Drawdowns (queda máxima), totais e por sequência de perdas  
- Quantos ganhos/perdas seguidos você já teve  
- Quanto tempo, em média, cada trade dura (no geral, só vencedores e só perdedores)  
- Estatísticas por dia da semana (melhor e pior dia)  
- Estatísticas por mês (melhor e pior mês)

### Como usar  

1. **Selecione “POST”**  
2. **Endereço:** `http://127.0.0.1:5000/api/tabela`  
3. **No corpo da requisição** escolha “form-data” e adicione:  
   - **Key:** `file`  
   - **Tipo:** File  
   - **Valor:** seu arquivo CSV  
4. Clique em **Send** e você verá um JSON com todas as métricas.

> **Se der erro de “nenhum arquivo enviado”**, verifique se o campo se chama exatamente `file`.  

---

## 2) Conversa com o ChatGPT  
**Rota:** `POST /chat`

### O que faz  
Encaminha qualquer conversa (lista de mensagens) para o modelo GPT-4 e te devolve a resposta dele.

### Como usar  

1. **Selecione “POST”**  
2. **Endereço:** `http://127.0.0.1:5000/chat`  
3. **No corpo da requisição**, passe um JSON assim:
```json
{
  "messages": [
    { "role": "system",    "content": "Você é um assistente amigável." },
    { "role": "user",      "content": "Oi, tudo bem?" }
  ]
}
