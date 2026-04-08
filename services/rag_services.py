import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.readers.file import PyMuPDFReader
import chromadb
from dotenv import dotenv_values

config = dotenv_values(".env")

loader = PyMuPDFReader()

class RAGService:
    """
    Uma classe de serviço para tarefas de Geração Aumentada por Recuperação (RAG).

    Esta classe encapsula a lógica para criar/carregar um índice vetorial
    a partir de documentos e consultá-lo para obter respostas contextualizadas de um LLM.
    """

    def __init__(self,
                 model_name='gemini-2.5-flash-lite',
                 embedding_model_name='intfloat/multilingual-e5-large', #'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 doc_path='./docs',
                 persist_path='./db/vector_db',
                 chunk_size=512,
                 chunk_overlap=50,
                 similarity_top_k=5,
                 similarity_cutoff=0.0):
        """
        Inicializa o serviço RAG com configurações personalizáveis.
        """

        self.chroma_path = "./db/chroma_db" 
        self.persist_path = persist_path
        self.doc_path = doc_path

        # Configurações globais do LlamaIndex
        Settings.llm = Settings.llm = GoogleGenAI(model=model_name, api_key=config['KEY']) # Ollama(model=model_name, request_timeout=120.0)
        Settings.embed_model = FastEmbedEmbedding(model_name=embedding_model_name)
        Settings.text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Inicializa o motor de busca
        self.query_engine = self._create_query_engine(
            similarity_top_k=similarity_top_k,
            similarity_cutoff=similarity_cutoff
        )

    def _load_or_create_index(self):
        """
        Carrega o índice vetorial do diretório de persistência se ele existir;
        caso contrário, o cria a partir dos documentos no doc_path.
        """
        
        db = chromadb.PersistentClient(path=self.chroma_path)
        chroma_collection = db.get_or_create_collection("futebol_mvp")
        
        # 2. Configura o Vector Store do Chroma
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 3. Lógica de Migração: Se o Chroma estiver vazio mas o VectorStore antigo existir
        if chroma_collection.count() == 0 and os.path.exists(self.persist_path):
            print("Migrando dados do SimpleVectorStore para ChromaDB (sem re-gerar embeddings)...")
            
            # Carrega o índice antigo (carrega na RAM só desta vez)
            old_storage_context = StorageContext.from_defaults(persist_dir=self.persist_path)
            old_index = load_index_from_storage(old_storage_context)
            
            # Recupera os nós com os embeddings já prontos
            nodes = list(old_index.docstore.docs.values())
            
            # Insere no novo índice vinculado ao Chroma
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            print("Migração concluída com sucesso!")
            return index

        # 4. Fluxo normal: Se já existe no Chroma, carrega direto (sem carregar tudo na RAM)
        if chroma_collection.count() > 0:
            print("Carregando índice diretamente do ChromaDB...")
            return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        # 5. Se for a primeira vez total: Cria do zero
        print("Criando novo índice do zero...")

        file_extractor = {".pdf": PyMuPDFReader()}

        documents = SimpleDirectoryReader(
            input_dir=self.doc_path,
            recursive=True,
            file_extractor=file_extractor
        ).load_data()

        # documents = SimpleDirectoryReader(self.doc_path).load_data()
        return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    def _create_query_engine(self, similarity_top_k, similarity_cutoff):
        """
        Cria e configura o motor de busca com recuperação.
        """
        index = self._load_or_create_index()

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

        synthesizer = get_response_synthesizer()

        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

    def query(self, query_text):
        """
        Executa uma consulta no pipeline RAG e retorna a resposta.
        """
        if not self.query_engine:
            raise RuntimeError("O motor de busca não foi inicializado.")
        
        response = self.query_engine.query(query_text)
        result = str(response)
        print(f"DEBUG: RAGService.query query: {query_text}...")
        print(f"DEBUG: RAGService.query returning: {result[:100]}...")
        return result
