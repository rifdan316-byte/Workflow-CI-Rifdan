"""
Proyek: Student Lifestyle and Stress Prediction
File: modelling_tuning.py (Kriteria 2 - Level Skilled)
Nama Mahasiswa: Rifdan
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # 1. Inisialisasi MLflow Tracking Lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # Sesuaikan port mlflow UI Anda
    mlflow.set_experiment("Student_Stress_Prediction_Rifdan")

    # 2. Memuat Dataset Hasil Preprocessing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "student_performance_preprocessing.csv")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset preprocessing tidak ditemukan di: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Pisahkan fitur dan target (Sesuaikan nama kolom target Anda, misal: 'Stress_Level')
    X = df.drop(columns=['Stress_Level'])
    y = df['Stress_Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Definisikan Ruang Parameter untuk Hyperparameter Tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    
    print("[INFO] Memulai Hyperparameter Tuning menggunakan GridSearchCV...")
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"[INFO] Parameter Terbaik: {best_params}")

    # 4. Manual Logging ke MLflow (Syarat Mutlak Level Skilled)
    with mlflow.start_run(run_name="Random_Forest_Tuning_Skilled"):
        
        # A. Log Hyperparameters secara manual
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Evaluasi Model pada Data Uji
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary') # sesuaikan jika target multiclass
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # B. Log Metrics secara manual (Meniru output autolog)
        mlflow.log_metric("training_accuracy", accuracy_score(y_train, best_model.predict(X_train)))
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        
        # C. Log Wujud Fisik Model ke dalam Tab Artifacts (Menjawab Komplain Reviewer)
        # Model akan disimpan ke dalam direktori bernama "best_rf_model"
        mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path="best_rf_model",
            registered_model_name="StudentStressRFModel"
        )
        
        print("[SUKSES] Eksperimen berhasil dicatat di MLflow dengan manual logging!")
        print(f"         Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()