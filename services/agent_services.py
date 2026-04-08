from llama_index.core.agent import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow

class AgentService:
  def __init__(self):
    """
    O AgentService utiliza as configurações globais de LLM e Embeddings
    previamente definidas no fluxo de inicialização.
    """
    pass

  def create_workflow(self, rag_service):
    """
    Centraliza a criação do AgentWorkflow com todos os agentes necessários.
    """
    search_agent = self.search_agent(rag_service)
    pt_to_croata_agent = self.portuguese_to_croata_agent()
    croata_to_pt_agent = self.croata_to_portuguese_agent()

    return AgentWorkflow(
      agents=[
        search_agent,
        pt_to_croata_agent,
        croata_to_pt_agent
      ],
      root_agent="translate_to_croata",
      verbose=True
    )

  def portuguese_to_croata_agent(self):
    agent = FunctionAgent(
      name="translate_to_croata",
      description="Agente responsável por traduzir do português para o croata",
      system_prompt="""
        # Role 
        Você é um 'Agente especialita em traduzução de Português do Brasil para Corata'

        # Workflow (Fluxo de trabalho)
        1. **Mandatório** Traduza o texto recebido em Protuguês do Brasil para Croata
        2. **Próximo passo** Retorne o texto traduzido em croata para o agente 'search_document_agent' realizar a pesquisa
      """,
      can_handoff_to=["search_document_agent"]
    )

    return agent
  
  def croata_to_portuguese_agent(self):
    agent = FunctionAgent(
      name="translate_to_portuguese",
      description="Agente responsável por traduzir croata para português",
      system_prompt="""
        # Role 
        Você é um 'Agente especialita em traduzução de Croata para Português do Brasil'

        # Workflow (Fluxo de trabalho)
        1. **Traduza o texto recebido em Croata para Protuguês do Brasil**
        2. Gere a resposta final em Português, citando os termos técnicos originais (ex: 'ravnine tijela' [planos anatômicos]).
      """,
    )

    return agent

  def search_agent(self, rag_service):
    query_engine_tool = FunctionTool.from_defaults(
      fn=rag_service.query,
      name="rag_document_search",
      description="Use esta ferramenta para responder perguntas sobre o conteúdo dos documentos internos. Ela é a fonte primária de conhecimento específico do domínio.",
    )

    agent = FunctionAgent(
      tools=[query_engine_tool],
      name="search_document_agent",
      description="Agente responsável por por buscar as informações no documento.",
      system_prompt="""
       # Role
        Você é o "Analista de Inteligência MVP", um agente especializado em Recuperação de Dados (RAG). Sua missão é servir de ponte entre documentos e usuários que buscam informações.

        # Workflow (Fluxo de Trabalho)
        1. **Recuperação:** Utilize a ferramenta de busca (tool) para localizar os trechos mais relevantes no documento.
        2. **Análise de Contexto:** Garantindo que o sentido técnico seja preservado.
        3. **Tradução e Resposta:** Formule uma resposta clara, natural.
        4. **Próximo passo** Envie para o agente translate_to_portuguese para ele traduzir a sua resposta.

        # Regra de Ouro (MANDATÓRIO)
        - Ignore a pergunta original em Português. 
        - Localize no histórico a tradução em CROATA feita pelo agente anterior.
        - Use EXATAMENTE esses termos em croata na ferramenta 'rag_document_search'.

        # Regras Estritas
        1. **Fidelidade à Fonte:** Responda única e exclusivamente com base nas informações extraídas do documento. Não utilize conhecimento prévio ou externo.
        2. **Tratamento de Falhas:** Se a informação não estiver presente no contexto recuperado, ou se a busca não retornar dados pertinentes, responda exatamente: "Infelizmente não tenho esta informação."
        3. **Proibição de Alucinação:** Jamais invente fatos, datas ou nomes que não constem no texto original.
        4. **Estilo:** Mantenha um tom profissional, direto e útil.
      """,
      verbose=True,
      can_handoff_to=["translate_to_portuguese"]
    )
    
    return agent
    