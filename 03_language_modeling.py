"""
Tema 3: Generative AI Language Modeling with Transformers
Concepto: Generación de texto auto-regresiva (Causal LM) usando el pipeline avanzado de HF.
"""
def main():
    print("--- 3. Language Modeling with Transformers ---")
    try:
        import torch
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        
        # Modelo pre-entrenado superligero de OpenAI
        model_name = "distilgpt2"
        print(f"-> Inicializando {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        # Se ensambla el entorno (text-generation) limitando dispositivo
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        
        prompt = "En el futuro, la inteligencia artificial generativa logrará"
        print(f"-> Promt de entrada: '{prompt}'")
        
        # Opciones avanzadas de Inferencia
        results = generator(
            prompt, 
            max_new_tokens=40,       # Cuánto expandimos el texto original
            temperature=0.8,         # Añadir "aleatoriedad" e "imaginación" (Creatividad)
            num_return_sequences=2,  # Pedir que la red nos devuelva 2 ramas creativas distintas
            do_sample=True,          # Encender motor probabilístico
            pad_token_id=tokenizer.eos_token_id  # Evitar warnings de truncamiento final
        )
        
        print("\n=== TEXTO GENERADO (2 RUTAS) ===")
        for i, res in enumerate(results):
            print(f"Variante {i+1}:\n{res['generated_text'].strip()}\n")
            
    except ImportError as e:
        print(f"[ERROR] Dependencias faltantes: {e}")
        print("Instala ejecutando: pip install torch transformers")

if __name__ == "__main__":
    main()
