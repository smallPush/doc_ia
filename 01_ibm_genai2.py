import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# ========================= CONFIGURACIÓN =========================
llm = ChatOllama(
    model="llama3.2:3b",      # cambia a "llama3.1:8b" si tienes más RAM/GPU
    temperature=0.7
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# ========================= 1. CARGAR Y INDEXAR DOCUMENTOS =========================
loader = PyPDFLoader("mi_documento.pdf")          # puedes poner varios PDFs
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db_rag_agent"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ========================= 2. CREAR EL TOOL DE RAG =========================
retriever_tool = create_retriever_tool(
    retriever,
    name="buscar_en_documentos",
    description="Busca información relevante en los documentos cargados. "
                "Úsalo siempre que la pregunta esté relacionada con el contenido de los PDFs."
)

tools = [retriever_tool]

# ========================= 3. MEMORIA PERSISTENTE =========================
memory = MemorySaver()   # esto es lo que hace la magia de la memoria persistente

# ========================= 4. CREAR EL AGENTE CON LANGGRAPH =========================
system_prompt = """Eres un asistente útil y preciso llamado Grok.
Usa el tool 'buscar_en_documentos' cuando la pregunta requiera información de los documentos.
Responde siempre de forma clara, amigable y en español.
Si no encuentras la información, dilo honestamente."""

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,               # ← memoria persistente
    prompt=system_prompt
)

# ========================= 5. FUNCIÓN PARA CHATEAR =========================
def chat_con_agentic_rag():
    print("🤖 Agentic RAG con LangGraph iniciado")
    print("Memoria persistente activada (puedes cerrar y volver a abrir)")
    print("Escribe 'salir' para terminar.\n")
    
    thread_id = "conversacion_1"   # cambia este ID para una nueva conversación
    
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() in ["salir", "exit", "adiós"]:
            print("¡Hasta luego! 👋")
            break
        
        # Configuración para memoria persistente
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invocar al agente
        respuesta = agent.invoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config
        )
        
        # Mostrar solo la respuesta final del agente
        print(f"Grok: {respuesta['messages'][-1].content}\n")

# ========================= EJECUTAR EL PROYECTO =========================
if __name__ == "__main__":
    chat_con_agentic_rag()