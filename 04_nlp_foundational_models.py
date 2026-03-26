"""
Tema 4: Gen AI Foundational Models for NLP & Language Understanding
Concepto: Uso de Modelos Fundacionales para múltiples tareas analíticas de Lenguaje Natural (PLN).
"""
def main():
    print("--- 4. Foundational Models for NLP ---")
    try:
        from transformers import pipeline
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        print("-> 1. Cargando pipeline de Análisis de Sentimiento (Sentiment-Analysis)...")
        sentiment = pipeline("sentiment-analysis", device=-1)
        # El modelo analiza si la frase es POSITIVE o NEGATIVE y da su grado de confianza (Score)
        print("Resultado Sentimiento:", sentiment("Me siento extremadamente satisfecho con esta aplicación!"))
        
        print("\n-> 2. Cargando pipeline de Reconocimiento de Entidades Nombradas (NER)...")
        # Identifica Organizaciones, Lugares, Personas... Agrupándolas (grouped_entities=True)
        ner = pipeline("ner", aggregation_strategy="simple", device=-1)
        print("Extracción NER:")
        for entidad in ner("Rubén fundó Microsoft Enterprise en Barcelona durante 1999."):
             print(f" - [{entidad['entity_group']}] detectado: {entidad['word']} (Certeza: {entidad['score']:.2f})")
        
        print("\n-> 3. Cargando pipeline Zero-Shot Classification...")
        # Capacidad de los LM Modernos: categorizar un texto SIN haber sido entrenado para esas categorías
        zero_shot = pipeline("zero-shot-classification", device=-1)
        texto = "El nuevo cohete reutilizable logró aterrizar en la plataforma marítima reduciendo costes un 30%."
        etiquetas = ["cocina", "astronomía/ciencia", "deportes", "economía"]
        
        print(f"Texto a analizar: '{texto}'")
        res_zs = zero_shot(texto, candidate_labels=etiquetas)
        print("\nZero-Shot (Categoría ganadora):", {res_zs["labels"][0]: res_zs["scores"][0]})
        print("Resto de probabilidades asignadas:", list(zip(res_zs['labels'][1:], res_zs['scores'][1:])))
        
    except ImportError as e:
         print(f"[ERROR] Dependencias faltantes: {e}")
         print("Instala ejecutando: pip install torch transformers sentencepiece")

if __name__ == "__main__":
    main()
