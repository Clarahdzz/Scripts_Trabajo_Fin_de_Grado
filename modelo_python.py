import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



# 0. OCULTAR WARNINGS DEL MLP


warnings.filterwarnings("ignore", category=ConvergenceWarning)


# 1. CONFIGURACIÓN GENERAL

DATASET_DIR = r"C:\Users\Emilio\Desktop\4_TELECO\TFG_EEG\valores_com_emotiv_128hz"

BAND_COLUMNS = [
    "T7_theta", "T7_mu", "T7_beta",
    "T8_theta", "T8_mu", "T8_beta",
    "AF3_theta", "AF3_alpha", "AF3_beta",
    "AF4_theta", "AF4_alpha", "AF4_beta",
    "Pz_alpha", "Pz_beta"
]

RAW_COLUMNS = ["AF3", "T7", "Pz", "T8", "AF4"]



# 2. EXTRAER ETIQUETA DESDE EL NOMBRE DEL ARCHIVO


def get_label_from_filename(filename):
    """
    Extrae la etiqueta únicamente según mano izquierda o mano derecha.
    Ignora si aparece abrir/cerrar en el nombre.
    """

    name = filename.lower()

    if "derecha" in name or "right" in name:
        return "derecha"

    if "izquierda" in name or "left" in name:
        return "izquierda"

    return "DESCONOCIDO"


# 3. EXTRAER FEATURES DE CADA CSV

def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Limpiar posibles espacios en nombres de columnas
    df.columns = [c.strip() for c in df.columns]

    features = {}

    
    # Features de bandas EEG
  
    for col in BAND_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values

            if len(values) > 0:
                features[f"{col}_mean"] = np.mean(values)
                features[f"{col}_std"] = np.std(values)
                features[f"{col}_min"] = np.min(values)
                features[f"{col}_max"] = np.max(values)
                features[f"{col}_median"] = np.median(values)

   
    # Features de señal EEG raw
   
    for col in RAW_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values

            if len(values) > 0:
                features[f"{col}_mean"] = np.mean(values)
                features[f"{col}_std"] = np.std(values)
                features[f"{col}_min"] = np.min(values)
                features[f"{col}_max"] = np.max(values)
                features[f"{col}_range"] = np.max(values) - np.min(values)
                features[f"{col}_median"] = np.median(values)


  
    # Features de lateralidad T7-T8
  
    eps = 1e-8

    if "T7_theta" in df.columns and "T8_theta" in df.columns:
        t7 = pd.to_numeric(df["T7_theta"], errors="coerce").mean()
        t8 = pd.to_numeric(df["T8_theta"], errors="coerce").mean()
        features["theta_diff_T7_T8"] = t7 - t8
        features["theta_ratio_T7_T8"] = t7 / (t8 + eps)
        features["theta_asym_T7_T8"] = (t7 - t8) / (t7 + t8 + eps)

    if "T7_mu" in df.columns and "T8_mu" in df.columns:
        t7 = pd.to_numeric(df["T7_mu"], errors="coerce").mean()
        t8 = pd.to_numeric(df["T8_mu"], errors="coerce").mean()
        features["mu_diff_T7_T8"] = t7 - t8
        features["mu_ratio_T7_T8"] = t7 / (t8 + eps)
        features["mu_asym_T7_T8"] = (t7 - t8) / (t7 + t8 + eps)

    if "T7_beta" in df.columns and "T8_beta" in df.columns:
        t7 = pd.to_numeric(df["T7_beta"], errors="coerce").mean()
        t8 = pd.to_numeric(df["T8_beta"], errors="coerce").mean()
        features["beta_diff_T7_T8"] = t7 - t8
        features["beta_ratio_T7_T8"] = t7 / (t8 + eps)
        features["beta_asym_T7_T8"] = (t7 - t8) / (t7 + t8 + eps)

   
    # Ratios mu/beta por canal
    
    if "T7_mu" in df.columns and "T7_beta" in df.columns:
        mu = pd.to_numeric(df["T7_mu"], errors="coerce").mean()
        beta = pd.to_numeric(df["T7_beta"], errors="coerce").mean()
        features["T7_mu_beta_ratio"] = mu / (beta + eps)

    if "T8_mu" in df.columns and "T8_beta" in df.columns:
        mu = pd.to_numeric(df["T8_mu"], errors="coerce").mean()
        beta = pd.to_numeric(df["T8_beta"], errors="coerce").mean()
        features["T8_mu_beta_ratio"] = mu / (beta + eps)

    return features


# 4. CARGAR DATASET


def load_dataset(dataset_dir):
    csv_files = [
    f for f in glob.glob(os.path.join(dataset_dir, "*.csv"))
    if os.path.basename(f).lower().startswith("abrir-cerrar_mano_")
]
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No se han encontrado CSV en: {dataset_dir}")

    rows = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        label = get_label_from_filename(filename)

        features = extract_features_from_csv(csv_path)

        if len(features) == 0:
            print(f"Archivo ignorado porque no tiene features válidas: {filename}")
            continue

        features["label"] = label
        features["file"] = filename
        features["path"] = csv_path

        rows.append(features)

    dataset = pd.DataFrame(rows)
    return dataset



# 5. FUNCIÓN PARA EVALUAR MODELOS

def evaluate_models(dataset, feature_cols, experiment_name):
    print("\n\n================================================")
    print(f"EXPERIMENTO: {experiment_name}")
    print("================================================")

    X = dataset[feature_cols].values
    y = dataset["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nClases codificadas:")
    for i, cls in enumerate(le.classes_):
        print(f"{i}: {cls}")

    print("\nNúmero de muestras:", X.shape[0])
    print("Número de features:", X.shape[1])

    # Definir validación cruzada
    class_counts = dataset["label"].value_counts()
    min_class_count = class_counts.min()

    if min_class_count < 2:
        raise ValueError("Hay alguna clase con menos de 2 muestras.")

    n_splits = min(5, min_class_count)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    # Modelos a comparar
    models = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000))
        ]),

        "SVM linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1))
        ]),

        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1, gamma="scale"))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),

        "MLP pequeño": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(10,),
                learning_rate_init=0.0001,
                max_iter=5000,
                random_state=42
            ))
        ]),

        "MLP 20-10": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(20, 10),
                learning_rate_init=0.0001,
                max_iter=5000,
                random_state=42
            ))
        ])
    }


    # Validación cruzada
    

    print("\n================================================")
    print("VALIDACIÓN CRUZADA")
    print("================================================")

    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y_encoded,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro"
            },
            return_train_score=True
        )

        results.append({
            "experiment": experiment_name,
            "model": name,
            "train_acc_mean": scores["train_accuracy"].mean(),
            "train_acc_std": scores["train_accuracy"].std(),
            "test_acc_mean": scores["test_accuracy"].mean(),
            "test_acc_std": scores["test_accuracy"].std(),
            "balanced_acc_mean": scores["test_balanced_accuracy"].mean(),
            "f1_macro_mean": scores["test_f1_macro"].mean()
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="test_acc_mean", ascending=False)

    print("\nResultados ordenados por accuracy:")
    print(results_df.to_string(index=False))

    
    # Optimización SVM RBF
    

    print("\n================================================")
    print("OPTIMIZACIÓN SVM RBF CON GRIDSEARCH")
    print("================================================")

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf"))
    ])

    param_grid = {
        "clf__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
        "clf__gamma": ["scale", "auto", 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    }

    grid = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X, y_encoded)

    print("Mejores parámetros:", grid.best_params_)
    print("Mejor balanced accuracy:", round(grid.best_score_, 4))

    resultados_grid = pd.DataFrame(grid.cv_results_)
    resultados_grid = resultados_grid.sort_values(by="mean_test_score", ascending=False)

    print("\nTop 10 SVM RBF:")
    print(resultados_grid[[
        "param_clf__C",
        "param_clf__gamma",
        "mean_train_score",
        "mean_test_score",
        "std_test_score"
    ]].head(10).to_string(index=False))

    
    # Train/test manual con mejor SVM
    

    print("\n================================================")
    print("TRAIN/TEST MANUAL CON MEJOR SVM")
    print("================================================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )

    best_svm = grid.best_estimator_
    best_svm.fit(X_train, y_train)
    y_pred = best_svm.predict(X_test)

    print("\nAccuracy test:", round(accuracy_score(y_test, y_pred), 4))
    print("Balanced accuracy test:", round(balanced_accuracy_score(y_test, y_pred), 4))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=le.classes_
    ))

    return results_df, resultados_grid



# 6. PROGRAMA PRINCIPAL

dataset = load_dataset(DATASET_DIR)

print("\n================================================")
print("DATASET CARGADO")
print("================================================")
print(dataset.head())

print("\nTamaño del dataset:")
print(dataset.shape)

print("\nEtiquetas detectadas:")
print(dataset["label"].value_counts())

# Comprobar archivos sin etiqueta izquierda/derecha
if "DESCONOCIDO" in dataset["label"].values:
    print("\n================================================")
    print("PROBLEMA: ARCHIVOS SIN ETIQUETA IZQUIERDA/DERECHA")
    print("================================================")
    print("Hay archivos cuyo nombre no permite detectar si son izquierda o derecha:")
    print(dataset[dataset["label"] == "DESCONOCIDO"]["file"].head(30).to_string(index=False))
    raise SystemExit

# Eliminar columnas metadata
metadata_cols = ["label", "file", "path"]


# EXPERIMENTO 1: TODAS LAS FEATURES

feature_cols_all = [col for col in dataset.columns if col not in metadata_cols]

results_all, grid_all = evaluate_models(
    dataset=dataset,
    feature_cols=feature_cols_all,
    experiment_name="Todas las features"
)



# EXPERIMENTO 2: SOLO T7/T8 + LATERALIDAD

selected_features = [
    # Bandas principales T7
    "T7_theta_mean",
    "T7_mu_mean",
    "T7_beta_mean",

    # Bandas principales T8
    "T8_theta_mean",
    "T8_mu_mean",
    "T8_beta_mean",

    # Desviaciones
    "T7_theta_std",
    "T7_mu_std",
    "T7_beta_std",
    "T8_theta_std",
    "T8_mu_std",
    "T8_beta_std",

    # Diferencias entre T7 y T8
    "theta_diff_T7_T8",
    "mu_diff_T7_T8",
    "beta_diff_T7_T8",

    # Ratios entre T7 y T8
    "theta_ratio_T7_T8",
    "mu_ratio_T7_T8",
    "beta_ratio_T7_T8",

    # Asimetrías entre T7 y T8
    "theta_asym_T7_T8",
    "mu_asym_T7_T8",
    "beta_asym_T7_T8",

    # Ratios mu/beta
    "T7_mu_beta_ratio",
    "T8_mu_beta_ratio"
]

# Nos quedamos solo con las que existan realmente
feature_cols_t7_t8 = [f for f in selected_features if f in dataset.columns]

results_t7t8, grid_t7t8 = evaluate_models(
    dataset=dataset,
    feature_cols=feature_cols_t7_t8,
    experiment_name="Solo T7/T8 + lateralidad"
)



# 7. GUARDAR RESULTADOS


output_features = os.path.join(DATASET_DIR, "dataset_features_extraidas.csv")
output_results_all = os.path.join(DATASET_DIR, "resultados_todas_features.csv")
output_grid_all = os.path.join(DATASET_DIR, "grid_svm_todas_features.csv")

output_results_t7t8 = os.path.join(DATASET_DIR, "resultados_t7t8_lateralidad.csv")
output_grid_t7t8 = os.path.join(DATASET_DIR, "grid_svm_t7t8_lateralidad.csv")

dataset.to_csv(output_features, index=False)
results_all.to_csv(output_results_all, index=False)
grid_all.to_csv(output_grid_all, index=False)

results_t7t8.to_csv(output_results_t7t8, index=False)
grid_t7t8.to_csv(output_grid_t7t8, index=False)

print("\n================================================")
print("ARCHIVOS GUARDADOS")
print("================================================")
print(output_features)
print(output_results_all)
print(output_grid_all)
print(output_results_t7t8)
print(output_grid_t7t8)

print("\n================================================")
print("FIN DEL ANÁLISIS")
print("================================================")
