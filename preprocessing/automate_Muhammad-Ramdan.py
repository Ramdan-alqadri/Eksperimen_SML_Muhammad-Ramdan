"""
automate_Muhammad-Ramdan.py

Script otomatisasi preprocessing dataset Titanic, mereplikasi persis
langkah-langkah yang dilakukan di Eksperimen_Muhammad-Ramdan.ipynb:

1. Handle missing value (Age -> median, Embarked -> mode, drop kolom Cabin)
2. Drop data duplikat
3. Drop kolom tidak relevan (PassengerId, Name, Ticket)
4. Handle outlier pada kolom Fare menggunakan metode IQR clipping
5. Encoding kolom kategorikal (Sex, Embarked) menggunakan LabelEncoder
6. Scaling kolom numerik (Age, Fare) menggunakan StandardScaler

Cara pakai:
    python automate_Muhammad-Ramdan.py

Input  : ../namadataset_raw/titanic_raw.csv
Output : namadataset_preprocessing/titanic_preprocessing.csv
"""

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load raw dataset dari CSV."""
    df = pd.read_csv(path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Age  : isi missing value dengan median
    - Embarked : isi missing value dengan mode (nilai paling sering muncul)
    - Cabin : dibuang karena >75% datanya kosong
    """
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"])
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris yang duplikat persis."""
    return df.drop_duplicates()


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Buang kolom yang tidak relevan untuk pemodelan."""
    cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def handle_outliers_iqr(df: pd.DataFrame, column: str = "Fare") -> pd.DataFrame:
    """
    Tangani outlier pada kolom numerik menggunakan metode IQR clipping:
    nilai di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR] di-clip ke batas tersebut.
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding kolom Sex dan Embarked menggunakan LabelEncoder."""
    df = df.copy()
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def scale_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Scaling kolom Age dan Fare menggunakan StandardScaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Jalankan seluruh pipeline preprocessing secara berurutan."""
    df = handle_missing_values(df)
    df = drop_duplicates(df)
    df = drop_irrelevant_columns(df)
    df = handle_outliers_iqr(df, column="Fare")
    df = encode_categorical(df)
    df = scale_numerical(df)
    return df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "..", "namadataset_raw", "titanic_raw.csv")
    output_dir = os.path.join(base_dir, "namadataset_preprocessing")
    output_path = os.path.join(output_dir, "titanic_preprocessing.csv")

    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_data(raw_path)
    df_clean = preprocess(df_raw)
    df_clean.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. Data disimpan di: {output_path}")
    print(f"Jumlah baris: {df_clean.shape[0]}, Jumlah kolom: {df_clean.shape[1]}")


if __name__ == "__main__":
    main()
