import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def get_data_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    possible_dirs = [
        os.path.join(script_dir, 'dataset_processed'),
        os.path.join(script_dir, 'data_processed'),
        os.path.join(repo_root, 'Eksperimen_SML_Rifdan', 'prepocessing', 'data_processed'),
        os.path.join(repo_root, 'prepocessing', 'data_processed'),
        os.path.join(repo_root, 'data_processed'),
        os.path.join(repo_root, 'dataset_processed'),
    ]
    for path in possible_dirs:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Folder dataset tidak ditemukan. Pastikan hasil preprocessing ada di salah satu: "
        "Eksperimen_SML_Rifdan/prepocessing/data_processed atau prepocessing/data_processed"
    )


def configure_mlflow():
    import mlflow
    import mlflow.sklearn

    # Saat dijalankan via `mlflow run`, MLflow sudah menyiapkan tracking URI dan
    # membuat run aktif. Jangan menimpa konfigurasi tersebut agar tidak konflik.
    if os.environ.get("MLFLOW_RUN_ID"):
        print("Dijalankan via `mlflow run`. Menggunakan tracking & run yang sudah aktif.")
        mlflow.autolog()
        return mlflow

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_mlflow_dir = os.path.join(repo_root, 'mlruns')
    remote_uri = "http://127.0.0.1:5000"

    mlflow.set_tracking_uri(remote_uri)
    try:
        mlflow.set_experiment("Student Stress Prediction Base")
        print(f"Terhubung ke MLflow server: {remote_uri}")
    except Exception:
        os.makedirs(local_mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(local_mlflow_dir)
        mlflow.set_experiment("Student Stress Prediction Base")
        print(f"Tidak dapat terhubung ke MLflow server {remote_uri}. Menggunakan penyimpanan lokal: {local_mlflow_dir}")

    mlflow.autolog()
    return mlflow


def train_base_model():
    # 1. Konfigurasi MLflow
    mlflow = configure_mlflow()
    
    # 2. Load Data Hasil Preprocessing
    base_data_dir = get_data_dir()

    X_train = pd.read_csv(os.path.join(base_data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(base_data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(base_data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(base_data_dir, 'y_test.csv')).values.ravel()
    
    # 3. Inisiasi Model Dasar Random Forest
    model = RandomForestClassifier(random_state=42)
    
    print("Memulai pelatihan model dasar Random Forest dengan MLflow Autolog...")

    def _train_and_log():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"\n=== Model Base Metrics ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, predictions)}")

    # Saat via `mlflow run` sudah ada run aktif; pakai run itu. Jika tidak, buat run baru.
    if mlflow.active_run() is not None:
        _train_and_log()
    else:
        with mlflow.start_run(run_name="Base_Random_Forest"):
            _train_and_log()

if __name__ == "__main__":
    train_base_model()