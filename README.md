# Soccer Multi-Agent RAG System

Este projeto é um assistente inteligente que utiliza uma arquitetura de multiagentes para realizar buscas semânticas em documentos técnicos. O sistema é otimizado para processar consultas em Português, realizar buscas em documentos (originalmente em Croata) e retornar respostas traduzidas e contextualizadas.

O projeto segue o princípio **KISS (Keep It Simple and Stupid)**, mantendo uma estrutura modular e direta.

## 🚀 Funcionalidades

- **RAG Multiagente:** Orquestração de agentes especializados em tradução e recuperação de dados.
- **Busca Semântica:** Utiliza ChromaDB e FastEmbed para encontrar informações precisas em documentos PDF.
- **Interface Amigável:** Chat interativo construído com Gradio.
- **Tradução Automática:** Pipeline integrado que traduz a intenção do usuário para o idioma do documento e vice-versa.

## 🛠️ Tecnologias

- **Framework:** [LlamaIndex](https://www.llamaindex.ai/)
- **LLM:** Google Gemini (via `google-genai`)
- **Embeddings:** `fastembed` (Modelo: `intfloat/multilingual-e5-large`)
- **Database:** `ChromaDB`
- **UI:** `Gradio`

## 📋 Pré-requisitos

1.  Python 3.10+
2.  Uma chave de API do Google Gemini.

## 🔧 Configuração

1.  **Clone o repositório:**
    ```bash
    git clone <url-do-repositorio>
    cd soccer
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure as variáveis de ambiente:**
    Crie um arquivo `.env` na raiz do projeto:
    ```env
    KEY=sua_chave_api_do_google_gemini
    ```

4.  **Adicione seus documentos:**
    Coloque os arquivos PDF que deseja indexar na pasta `docs/`.

## 🏃 Como Executar

Para iniciar a aplicação e abrir a interface do Gradio:

```bash
python app.py
```

Acesse o link local (ex: `http://127.0.0.1:7860`) que aparecerá no terminal.

## 🏗️ Estrutura do Projeto

- `app.py`: Ponto de entrada da aplicação e interface UI.
- `services/`:
  - `rag_services.py`: Lógica de indexação, persistência no ChromaDB e motor de busca.
  - `agent_services.py`: Definição dos agentes (Tradução e Busca) e do fluxo de trabalho (`AgentWorkflow`).
- `docs/`: Diretório para armazenamento dos documentos PDF.
- `db/`: Armazenamento local do banco de dados vetorial.

## 🧠 Padrão de Design: KISS

Este projeto foi desenvolvido focando na simplicidade:
- **Modularidade:** Separação clara entre serviços de dados (RAG) e serviços de inteligência (Agentes).
- **Sem Overengineering:** Uso de ferramentas robustas que resolvem o problema com o mínimo de código customizado.
- **Configuração Única:** Centralização de configurações globais no LlamaIndex Settings.
