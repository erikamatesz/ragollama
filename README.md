# Construindo um RAG local com Python e Ollama

<p align="center">
  <img src="https://alpacaglobalstore.com/cdn/shop/articles/foto_blog_sostenible-3.jpg?v=1738071396&width=3000" alt="Lhamas em campo" width="500px">
</p>


Este repositório foi criado como apoio para a palestra no meetup presencial do PythOnRio.

## Llama Del Rey

Para demonstrar na prática, usamos um cenário interno da (fictícia) **Llama Land**: uma assistente de IA que ajuda colaboradores a encontrar informações nos documentos da empresa.  

O objetivo aqui é construir uma **prova de conceito (PoC)** de **RAG local** usando **Python + Ollama + FAISS**.

**O que esta PoC valida**
- Recuperar trechos relevantes dos documentos e citar as fontes.
- Gerar respostas contextualizadas a partir desses trechos.
- Rodar 100% local (sem depender de nuvem) com modelos do Ollama.

**Escopo**
- **index.py**: Indexação de PDFs/CSVs em `./docs` → FAISS.
- **app.py**: API simples (`/ask`) que recebe a pergunta, busca os chunks e consulta o LLM.
- **.env.example**: Configuração mínima via `.env`. Crie seu `.env` a partir do exemplo.

### Futuro (fictício) desta PoC

Se fosse um projeto real dentro de uma empresa, esta solução de RAG local poderia evoluir para:

- **Implantação em nuvem**: hospedado em servidores corporativos ou infraestrutura em nuvem (AWS, Azure, GCP) para acesso por toda a equipe.
- **Base de conhecimento expandida**: incluir documentos de diversos setores, repositórios de código, wikis internas e relatórios.
- **Atualização contínua**: pipeline automático para indexar novos documentos ou alterações sem precisar rodar a indexação manualmente.
- **Segurança e permissões**: controle de acesso por usuário ou área, garantindo que cada colaborador veja apenas informações que pode acessar.
- **Interface de usuário amigável**: portal web ou integração com ferramentas internas (Slack, Teams, intranet).
- **Monitoramento e métricas**: acompanhar uso, latência, precisão das respostas e satisfação dos usuários.

## Como executar essa PoC localmente

### Pré-requisitos

1. Ollama instalado e com os modelos `nomic-embed-text:latest` e `gemma3:4b` baixados.
2. Python 3.10 ou superior.

**Importante:** Existem modelos mais eficientes que o `gemma3:4b`, porém, nem toda máquina aguenta ;)

### Setup

1. Crie e ative um `venv`.
2. Execute `pip install -r requirements.txt`.
3. Rode o Ollama com o comando `ollama serve`.
4. Rode a aplicação com o comando `uvicorn app:app --reload --host 127.0.0.1 --port 8000`.

Para acessar a documentação da API e realizar testes: http://localhost:8000/docs
