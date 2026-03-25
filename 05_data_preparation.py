"""
Tema 5: Generative AI and LLMs: Architecture and Data Preparation
Concepto: Extracción, Limpieza, Filtrado y Tokenización Masiva usando librerías industriales (Datasets).
"""
def main():
    print("--- 5. Architecture and Data Preparation ---")
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        print("-> Descargando dataset (Noticias de AgNews - Solo 50 muestras)...")
        # Datasets permite cargar gigabytes de modo perezoso (Lazy Loading). Aquí usamos un trozo.
        dataset = load_dataset("ag_news", split="train[:50]")
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # 1. Pipeline de Filtrado: Quitamos noticias que sean demasiado cortas
        # Normalmente un modelo tira basura si se le da basura ("Garbage In, Garbage Out")
        filtered_ds = dataset.filter(lambda record: len(record['text']) > 50)
        print(f"Limpieza completada. Datos validados: {len(filtered_ds)} (Originales antes de filtrar: {len(dataset)})")
        
        # 2. Pipeline de Transformación de Texto a Tensores
        def tokenize_batch(batch):
            # padding="max_length" iguala todas las listas de tokens con "ceros" para meterlos a la gráfica en matrices cuadradas.
            # truncation=True corta los excesos de palabras si superan 64 para no explotar la memoria RAM.
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
            
        print("\n-> Ejecutando Map/Reduce y Tokenización Vectorizada...")
        tokenized_ds = filtered_ds.map(tokenize_batch, batched=True)
        
        # Ajustamos los tipos resultantes para que los digiera PyTorch en vez de dejarlos en listas de Python
        tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        print("\n=== AUDITORÍA ARQUITECTÓNICA (Muestra [0]) ===")
        print("Noticia Original:\n", filtered_ds[0]["text"][:100], "...")
        print("\nMatriz Input IDs (Palabras en números algebraicos):")
        print(tokenized_ds[0]["input_ids"].shape, "->", tokenized_ds[0]["input_ids"][:15])
        print("Máscara de Atención (1=Token Real, 0=Padding artificial):")
        print(tokenized_ds[0]["attention_mask"].shape, "->", tokenized_ds[0]["attention_mask"][:15])
        print("==============================================")
        
    except ImportError as e:
         print(f"[ERROR] Dependencias faltantes: {e}")
         print("Instala ejecutando: pip install transformers datasets torch")

if __name__ == "__main__":
    main()
