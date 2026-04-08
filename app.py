from services.rag_services import RAGService
from services.agent_services import AgentService
import asyncio
import gradio as gr
from dotenv import dotenv_values

config = dotenv_values(".env")


async def main():
    """
    Função principal para executar a aplicação de chat RAG seguindo o padrão KISS.
    """
    print("=== Assistente de Futebol (RAG) ===")
    print("Carregando serviços, por favor aguarde...")
    
    try:
        # Inicializa os serviços de RAG e Agentes
        rag_service = RAGService()
        agent_service = AgentService()
        
        # Cria o workflow centralizado com todos os agentes
        workflow = agent_service.create_workflow(rag_service)

        async def chat_with_bot(message, chat_history):
            if chat_history is None:
                chat_history = []

            response = await workflow.run(user_msg=message)

            chat_history.append({"role":"user", "content": message})
            chat_history.append({"role":"assistant", "content": str(response)})

            return "", chat_history

        def reset_chat():
            # workflow.reset()
            return []

        with gr.Blocks() as interface:

            gr.Markdown('# Multiagentes de tradução')
            chatbot = gr.Chatbot(label="Multiagentes de tradução")
            msg = gr.Textbox(label='Digite sua pergunta')
            limpar = gr.Button('Limpar conversa')
            msg.submit(chat_with_bot, [msg, chatbot], [msg, chatbot])
            limpar.click(reset_chat, None, [chatbot], queue=False)

        interface.launch(debug=True)


    except Exception as e:
        print(f"Erro crítico ao iniciar a aplicação: {e}")


    

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAplicação encerrada pelo usuário.")
