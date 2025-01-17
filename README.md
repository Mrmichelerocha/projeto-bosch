# Projeto Bosch

## Objetivo principal:

Construir um sistema que permita carregar documentos (PDF, Word ou texto), extrair deles as informações principais (texto, tabelas, imagens), organizar essas informações em “chunks” e responder a perguntas do usuário usando um modelo de linguagem (LLM) que consulta os dados fornecidos pelos documentos.

Este repositório tem como objetivo construir um sistema que permita:

- **Carregar documentos** (PDF, Word ou texto)
- **Extrair informações principais** (texto, tabelas, imagens)
- **Organizar informações em “chunks”**
- **Responder perguntas do usuário** utilizando um modelo de linguagem (LLM) que consulta os dados fornecidos pelos documentos

## Escopo Atual:

- **Formato de documentos**: Apenas PDF (em versões futuras, planeja-se suportar Word e arquivos de texto).
- **Interface**: Linha de comando (CLI) ou Sistema Web Simples por ora.

---

## Arquivo enviado:
- Relatório de Vendas Mensais Bosch.pdf

## Exemplos de Perguntas:

### Para o texto:
- Quais foram os fatores principais que contribuíram para o crescimento significativo nas vendas dos produtos A e B no último trimestre?
- De que forma a revisão das práticas de atendimento ao cliente e a implementação de ferramentas de CRM impactaram nas vendas recorrentes?

### Para a tabela:
- Qual é o total de vendas por produto?
- Qual foi o ano com maior venda total?
- Quais são as vendas por mês?

### Para a imagem:
- Quais características da Furadeira/Parafusadeira a Bateria 12V GSR120-LI tornam seu uso mais prático e eficiente para trabalhos em locais com pouca iluminação?
- De que forma o design função de impacto da Furadeira de Impacto Bosch GSB 450 RE 450W facilita o uso em materiais mais resistentes como concreto?


# Como Executar o codigo:

- Crie uma virtualenv `virtualenv .venv` saiba mais em documentação https://docs.python.org/3/library/venv.html
- Ative a virtualenv `.venv\Scripts\activate` saiba mais em documentação https://docs.python.org/3/library/venv.html
- Instale as dependencias do ambiente `pip install -r  requirements.txt` estou usando python 3.10.0
- Rode o script app.py `Python app.py`


![Captura de tela 2025-01-17 195023](https://github.com/user-attachments/assets/c8841231-9f8b-46eb-9bfc-00238bce0f42)

