import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def main():
    # 1. Set nama Eksperimen di MLflow
    # mlflow.set_experiment("Eksperimen_Baseline_Rifdan")
    
    # 2. Mengaktifkan Autologging otomatis dari MLflow
    mlflow.sklearn.autolog()
    
    # 3. Membaca dataset hasil preprocessing
    X_train = pd.read_csv('student-lifestyle-preprocessing/X_train.csv')
    X_test = pd.read_csv('student-lifestyle-preprocessing/X_test.csv')
    y_train = pd.read_csv('student-lifestyle-preprocessing/y_train.csv').values.ravel()
    y_test = pd.read_csv('student-lifestyle-preprocessing/y_test.csv').values.ravel()
    
    # 4. Inisialisasi dan Pelatihan Model Baseline
    print("--- Melatih Baseline Model dengan Autolog ---")
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi komponen uji
        predictions = model.predict(X_test)
        print("Evaluasi Model Selesai. Silakan cek MLflow UI.")

if __name__ == "__main__":
    main()