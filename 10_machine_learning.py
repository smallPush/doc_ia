"""
Tema 10: Machine Learning with Python
Concepto: Creación, Evaluación y Entrenamiento de un 'Clasificador Random Forest' de predicción médica.
"""
def main():
    print("--- 10. Ecosistema Clásico de Machine Learning (Scikit-Learn) ---")
    try:
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        import pandas as pd
        import numpy as np

        print("\n-> 1. Extracción de Datos (Carga del Dataframe de Wisconsin Breast Cancer)...")
        dataset = load_breast_cancer()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        
        print("\n=== MUESTRA DEL DATASET DE ENTRADA (Múltiples biomarcadores) ===")
        print(df.head(2).to_string())
        
        # 'X' contiene características (tamaño tumor, densidad), 'Y' es la clase (Maligno/Benigno).
        X = dataset.data
        y = dataset.target
        print(f"\nDistribucción Vectorial: {X.shape[0]} Pacientes | {X.shape[1]} Variables.")

        print("\n-> 2. División de datos (Train vs Test)...") # Prevenimos que el modelo "memorice" en lugar de "aprender"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\n-> 3. Escalado Standardizado (Z-Score Normalization)...") 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("\n-> 4. Entrenamiento del Algoritmo Predictivo (RandomForestClassifier)...")
        # El Bosque Aleatorio usa en este caso 100 árboles de decisión diferentes
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train_scaled, y_train)

        print("\n-> 5. Predicción a Ciegas sobre pacientes nunca vistos y Evaluación de Robustez...")
        predicciones = clf.predict(X_test_scaled)
        
        precision_exacta = accuracy_score(y_test, predicciones)
        print(f"\n[OK] El modelo acertó en un {precision_exacta * 100:.2f}% de los casos.")
        
        print("\n=== REPORTE MÉDICO/TÉCNICO (Sensibilidad y Precisión) ===")
        print(classification_report(y_test, predicciones, target_names=dataset.target_names))

    except ImportError as e:
        print(f"\n[ERROR] El Stack analítico está incompleto: {e}")
        print("Corre esto en tu terminal: pip install scikit-learn pandas numpy")

if __name__ == "__main__":
    main()
