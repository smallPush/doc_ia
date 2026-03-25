"""
Tema 11: Introduction to Deep Learning & Neural Networks with Keras
Concepto: Arquitectura de Red Neuronal Profunda Secuencial (DNN) construida usando la abstracción Keras (Backend TF).
"""
def main():
    print("--- 11. Redes Neuronales Artificiales Funcionales (Keras) ---")
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        
        # Desactivamos advertencias agresivas de Cuda/Numa en consola
        tf.get_logger().setLevel('ERROR')
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        print("\n-> 1. Generación Artificial Numérica (Mock Data)...")
        # Matriz: 1,500 ejemplos, 20 dimensiones/rasgos cada uno.
        x_train = np.random.random((1500, 20))
        # Clasificación binaria (Respuestas 0 o 1)
        y_train = np.random.randint(2, size=(1500, 1))

        print("\n-> 2. Apilamiento Topológico (DNN Sequential)...")
        # Agregamos las neuronas (Múltiples "Capas Densas o Completamente Conectadas")
        model = Sequential([
            Dense(64, activation='relu', input_dim=20, name="Capa_Entrada"),  # 64 Neuronas Iniciales, Relu evita que desaparezca el gradiente matemáticamente
            Dropout(0.5, name="Capa_Reguladora"),                             # "Rompe/Apaga" 50% de las neuronas temporalmente para que el modelo generalice y no memorice
            Dense(64, activation='relu', name="Capa_Oculta_1"),               
            Dense(32, activation='relu', name="Capa_Oculta_2"),
            Dense(1, activation='sigmoid', name="Capa_Clasificacion_Salida")  # Sigmoid aplasta el resultado a un %, p. ej: si >0.5 es Clase 1.
        ])
        
        print("\n-> 3. Compilador del Plan de Aprendizaje y Pérdida...")
        model.compile(optimizer='adam',                   # Algoritmo estrella para empujar gradientes de forma adaptativa
                      loss='binary_crossentropy',         # Fórmula penalizadora binaria (El "Castigo" en la función coste)
                      metrics=['accuracy'])               # Qué queremos rastrear para humanos
        
        print("\n=== MAPA DE RECURSOS DE LA RED ===")
        model.summary()

        print("\n-> 4. Empezando a Entrenar (Cálculos hacia atrás / Backpropagation)...")
        # batch_size=32 (actualiza las métricas cada 32 ejemplos). epochs=10 (recorre las 1500 muestras 10 veces).
        historial = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
        
        print("\n-> 5. Rendimiento en Validation Set...")
        final_loss = historial.history['val_loss'][-1]
        final_acc = historial.history['val_accuracy'][-1]
        
        print(f"\n[Terminado] Pérdida Funcional de validación Base: {final_loss:.4f} | Precisión contra Val: {final_acc * 100:.2f}%")
        
    except ImportError as e:
         print(f"\\n[ERROR] Dependencias faltantes: {e}")
         print("Instala ejecutando: pip install tensorflow numpy")

if __name__ == "__main__":
    main()
