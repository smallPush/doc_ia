"""
Tema 7: Project: Generative AI Applications with RAG and LangChain
Concepto: Implementación "End-to-End" de Recuperación Aumentada (RAG) combinando FAISS y modelos abiertos.
"""
def main():
    print("--- 7. RAG (Retrieval-Augmented Generation) con LangChain ---")
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_classic.chains import RetrievalQA
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        import os
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        print("-> Fase 0. Fabricando documento secreto de la empresa (Base de Datos Viva)...")
        os.makedirs("07_rag_langchain", exist_ok=True)

        # Load a pdf
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader("07_rag_langchain/07_rag_langchain.pdf")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(docs)

        print("-> Fase 2. Representación y Vectorización Matemática (Embeddings)...")
        # Convierte el PDF/Texto en coordenadas numéricas para hacer búsquedas matemáticas puras
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # FAISS es la Base de Datos Vectorial de Meta. Se carga en RAM al instante.
        vectorstore = FAISS.from_documents(docs, embeddings)

        print("-> Fase 3. Instanciador del Cerebro Central (LLM)...")
        # GPT2 es un modelo de generación de texto probado y compatible
        pipe = pipeline("text-generation", model="gpt2", max_new_tokens=100, device=-1)
        llm = HuggingFacePipeline(pipeline=pipe)

        print("-> Fase 4. Orquestación RAG (Retriever + LLM)...")
        # Entrelazamos la búsqueda en VectorStore con el prompt de contexto hacia el LLM.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" = Meter directamente los documentos encontrados al prompt.
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}) # Traer los mejores 2 fragmentos geométricos.
        )

        pregunta = "What is CTTI?"
        print(f"\n=== INPUT DEL USUARIO ===\n'{pregunta}'")
        print("\n(Analizando distancias del vector, recuperando memoria semántica...)")
        respuesta = qa_chain.invoke(pregunta)

        print("\n=== RESPUESTA SISTEMA RAG INTEGRADOR ===")
        print(respuesta["result"])

        if os.path.exists("07_rag_langchain/conocimiento_rag.txt"):
            os.remove("07_rag_langchain/conocimiento_rag.txt") # Limpiamos el rastro

    except ImportError as e:
         print(f"[ERROR] Te faltan engranajes: {e}")
         print("Instala ejecutando: pip install langchain langchain-community faiss-cpu sentence-transformers transformers")

if __name__ == "__main__":
    main()
