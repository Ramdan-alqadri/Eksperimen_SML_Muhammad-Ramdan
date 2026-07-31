"""
modelling.py

Script pelatihan model untuk Kriteria 2 (Membangun Model Machine Learning).
Melatih model RandomForestClassifier pada dataset Titanic yang sudah
melalui tahap preprocessing (Kriteria 1), dan mencatat seluruh proses
pelatihan (parameter, metrik, artefak model) secara otomatis ke MLflow
menggunakan mlflow.sklearn.autolog().

Input  : preprocessing/namadataset_preprocessing/titanic_preprocessing.csv
Output : Run baru di MLflow Tracking (default: http://127.0.0.1:5000)

Cara pakai:
    1. Jalankan MLflow Tracking Server terlebih dahulu (di terminal terpisah):
         mlflow server --host 127.0.0.1 --port 5000
       (atau `mlflow ui` kalau hanya ingin tracking lokal biasa)

    2. Jalankan script ini:
         python modelling.py
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_preprocessed_data() -> pd.DataFrame:
    """Load dataset Titanic yang sudah dipreprocessing (hasil Kriteria 1)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir,
        "preprocessing",
        "namadataset_preprocessing",
        "titanic_preprocessing.csv",
    )
    return pd.read_csv(data_path)


def main():
    # 1. Set tracking URI ke MLflow server lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Titanic_RandomForest")

    # 2. Load data hasil preprocessing
    df = load_preprocessed_data()
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # 3. Split data train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Aktifkan autolog MLflow (mencatat parameter, metrik, dan model
    #    secara otomatis tanpa perlu mlflow.log_param/log_metric manual)
    mlflow.sklearn.autolog()

    # 5. Training model & logging berjalan otomatis di dalam run ini
    with mlflow.start_run(run_name="RandomForest_Basic"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Training selesai. Accuracy pada test set: {acc:.4f}")


if __name__ == "__main__":
    main()
