"""
Tema 7: Project: Generative AI Applications with RAG and LangChain
Concepto: Implementación "End-to-End" de Recuperación Aumentada (RAG) combinando FAISS y modelos abiertos.
"""

def main():
    print("--- 7. RAG (Retrieval-Augmented Generation) con LangChain ---")
    try:
        import os
        import logging

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_classic.chains import RetrievalQA
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        print("-> Fase 0. Fabricando documento secreto de la empresa (Base de Datos Viva)...")
        os.makedirs("07_rag_langchain", exist_ok=True)

        pdf_path = "07_rag_langchain/07_rag_langchain.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"No se encontró el PDF en: {pdf_path}. Coloca el documento allí."
            )

        print("-> Fase 1. Carga de documentos (PDF)...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        print("-> Fase 2. Split del documento en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(docs)

        print("-> Fase 3. Representación y Vectorización Matemática (Embeddings)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # === Selección de modelo + explicación corta ===
        print("\nModelos disponibles para el LLM:\n")

        print("  1) gpt2")
        print("     - Modelo clásico (~124M parámetros). Pequeño, rápido, generalista,")
        print("       pero con peor comprensión y razonamiento que modelos modernos.")  # [web:29][web:38][web:41]

        print("\n  2) microsoft/Phi-3-mini-4k-instruct")
        print("     - Modelo moderno (~3.8B parámetros), entrenado para seguir instrucciones")
        print("       y con muy buen razonamiento para su tamaño, ideal para RAG ligero.")  # [web:21][web:24][web:39]

        print("\n  3) projecte-aina/aguila-7b")
        print("     - Modelo de 7B parámetros, especializado en catalán, español e inglés;")
        print("       más pesado pero muy fuerte para contenido multilingüe local.")  # [web:28][web:37][web:40]

        opcion = input("\nElige modelo [1/2/3] (por defecto 1): ").strip() or "1"

        if opcion == "1":
            model_name = "gpt2"
        elif opcion == "2":
            model_name = "microsoft/Phi-3-mini-4k-instruct"
        elif opcion == "3":
            model_name = "projecte-aina/aguila-7b"
        else:
            print("Opción no reconocida, usando gpt2 por defecto.")
            model_name = "gpt2"

        print(f"\n-> Fase 4. Instanciador del Cerebro Central (LLM = {model_name})...")

        pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=200,
            device=-1,  # CPU; cambia a 0 si tienes GPU
        )
        llm = HuggingFacePipeline(pipeline=pipe)

        print("-> Fase 5. Orquestación RAG (Retriever + LLM)...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=False,
        )

        pregunta = input("\nEscribe tu pregunta sobre el PDF: ")
        if not pregunta.strip():
            pregunta = "What is CTTI?"

        print(f"\n=== INPUT DEL USUARIO ===\n'{pregunta}'")
        print("\n(Analizando distancias del vector, recuperando memoria semántica...)")

        respuesta = qa_chain.invoke({"query": pregunta})

        print("\n=== RESPUESTA SISTEMA RAG INTEGRADOR ===")
        if isinstance(respuesta, dict):
            print(respuesta.get("result") or respuesta.get("answer") or respuesta)
        else:
            print(respuesta)

    except ImportError as e:
        print(f"[ERROR] Te faltan engranajes: {e}")
        print(
            "Instala ejecutando: "
            "pip install langchain langchain-community langchain-text-splitters "
            "faiss-cpu sentence-transformers transformers"
        )
    except Exception as e:
        print(f"[ERROR] Algo ha fallado en la demo RAG: {e}")


if __name__ == "__main__":
    main()