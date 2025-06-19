import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from joblib import dump
import sys
import os

def preprocess_data(data, target_column, save_path, header_path):
    # Menentukan fitur numerik dan kategoris
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()
    column_names = data.columns.drop(target_column)
    
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(header_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {header_path}")

    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Simpan data variabel X ke data.csv sebelum splitting
    X.to_csv(os.path.join(os.path.dirname(header_path), 'data.csv'), index=False)
    print(f"Data variabel X berhasil disimpan ke: {os.path.join(os.path.dirname(header_path), 'data.csv')}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    dump(preprocessor, save_path)
    print(f"Preprocessor berhasil disimpan ke: {save_path}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # CLI: python automate.py <data_path> <target_column> <save_path> <header_path>
    if len(sys.argv) == 5:
        data_path = sys.argv[1]
        target_column = sys.argv[2]
        save_path = sys.argv[3]
        header_path = sys.argv[4]
        data = pd.read_csv(data_path)
        X_train, X_test, y_train, y_test = preprocess_data(data, target_column, save_path, header_path)
        
        # Membuat direktori output bernama 'heart_automate_output' di 'heart_preprocessing' jika belum ada
        outdir = os.path.join(os.path.dirname(header_path), 'heart_automate_output')
        os.makedirs(outdir, exist_ok=True)

        # Simpan langsung ke CSV
        pd.DataFrame(X_train).to_csv(os.path.join(outdir, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(outdir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(outdir, "y_train.csv"), index=False, header=[target_column])
        pd.DataFrame(y_test).to_csv(os.path.join(outdir, "y_test.csv"), index=False, header=[target_column])
    else:
        print("Usage: python automate_Filza-Rahma-Muflihah.py <data_path> <target_column> <save_path> <header_path>")
