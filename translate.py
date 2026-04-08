from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, SimpleDirectoryReader
from dotenv import load_dotenv
import asyncio
import os

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do LLM
Settings.llm = Ollama(model='llama3.2:latest', request_timeout=120.0)

# Função para salvar no arquivo
def add_lines(text: str):
    """
    Salva o texto traduzido no arquivo markdown.
    """
    os.makedirs('./translate', exist_ok=True)
    try:
        with open('./translate/doc.md', 'a', encoding='utf-8') as file:
            file.write(text + '\n')
    except Exception as e:
        print(f"Erro ao salvar no arquivo: {e}")

# Configuração do Splitter (overlap=0 para evitar duplicidade)
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=500
)

# Leitura dos documentos
file_extractor = {".pdf": PyMuPDFReader()}
documents = SimpleDirectoryReader(
    input_dir='./docs',
    recursive=True,
    file_extractor=file_extractor
).load_data()

nodes = splitter.get_nodes_from_documents(documents)

async def translate_all():
    """
    Itera pelos nós e traduz usando a LLM diretamente.
    """
    for node in nodes:
        prompt = (
            "Traduza o seguinte texto do Croata para Português do Brasil mantendo o tom técnico. "
            "Retorne APENAS a tradução em português, sem comentários ou explicações e ignore número da página, cabeçalhos e imagens:\n\n"
            f"{node.text}"
        )
        try:
            response = await Settings.llm.acomplete(prompt)
            add_lines(str(response))
        except Exception as e:
            print(f"Erro na tradução do bloco: {e}")

if __name__ == "__main__":
    asyncio.run(translate_all())
