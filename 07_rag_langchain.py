"""
Tema 7: Project: Generative AI Applications with RAG and LangChain
Concepto: Implementación "End-to-End" de Recuperación Aumentada (RAG) combinando FAISS y modelos abiertos.
"""
def main():
    print("--- 7. RAG (Retrieval-Augmented Generation) con LangChain ---")
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import CharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        import os
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        
        print("-> Fase 0. Fabricando documento secreto de la empresa (Base de Datos Viva)...")
        os.makedirs("07_rag_langchain", exist_ok=True)
        with open("07_rag_langchain/conocimiento_rag.txt", "w", encoding="utf-8") as f:
            f.write("ProjectZero es un sistema clasificado desarrollado en 2026. Sus fundadores son Rubén y su IA. "
                    "El objetivo del proyecto es colonizar el mercado tecnológico mediante automatización generativa.\n"
                    "El presupuesto corporativo fue de 3 billones de créditos marcianos y durará 2 años.")

        print("-> Fase 1. ETL: Ingesta del Documento y Fragmentación (Chunking)...")
        # El Chunking evita pasar el límite de Tokens del modelo LLM. 
        loader = TextLoader("07_rag_langchain/conocimiento_rag.txt", encoding="utf-8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(separator=".", chunk_size=200, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        
        print("-> Fase 2. Representación y Vectorización Matemática (Embeddings)...")
        # Convierte el PDF/Texto en coordenadas numéricas para hacer búsquedas matemáticas puras
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        # FAISS es la Base de Datos Vectorial de Meta. Se carga en RAM al instante.
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        print("-> Fase 3. Instanciador del Cerebro Central (LLM)...")
        # FLAN-T5 es experto en resumir y buscar información puntual
        pipe = pipeline("text2text-generation", model="google/flan-t5-large", max_length=100)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print("-> Fase 4. Orquestación RAG (Retriever + LLM)...")
        # Entrelazamos la búsqueda en VectorStore con el prompt de contexto hacia el LLM.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", # "stuff" = Meter directamente los documentos encontrados al prompt.
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}) # Traer los mejores 2 fragmentos geométricos.
        )
        
        pregunta = "¿Quiénes son los fundadores de ProjectZero, cuánto dinero costó y de qué va?"
        print(f"\n=== INPUT DEL USUARIO ===\n'{pregunta}'")
        print("\n(Analizando distancias del vector, recuperando memoria semántica...)")
        respuesta = qa_chain.invoke(pregunta)
        
        print("\n=== RESPUESTA SISTEMA RAG INTEGRADOR ===")
        print(respuesta["result"])
        
        os.remove("07_rag_langchain/conocimiento_rag.txt") # Limpiamos el rastro
        
    except ImportError as e:
         print(f"[ERROR] Te faltan engranajes: {e}")
         print("Instala ejecutando: pip install langchain langchain-community faiss-cpu sentence-transformers transformers")

if __name__ == "__main__":
    main()
