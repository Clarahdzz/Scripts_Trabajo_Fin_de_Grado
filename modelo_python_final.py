import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATASET_DIR = r"C:\Users\Emilio\Desktop\4_TELECO\TFG_EEG\valores_com_emotiv_128hz"

BAND_COLUMNS = [
    "T7_theta", "T7_mu", "T7_beta",
    "T8_theta", "T8_mu", "T8_beta",
    "AF3_theta", "AF3_alpha", "AF3_beta",
    "AF4_theta", "AF4_alpha", "AF4_beta",
    "Pz_alpha", "Pz_beta"
]

RAW_COLUMNS = ["AF3", "T7", "Pz", "T8", "AF4"]


# ============================================================
# 2. ETIQUETA DESDE NOMBRE DEL ARCHIVO
# ============================================================

def get_label_from_filename(filename):
    name = filename.lower()

    if "derecha" in name or "right" in name:
        return "derecha"

    if "izquierda" in name or "left" in name:
        return "izquierda"

    return "DESCONOCIDO"


# ============================================================
# 3. EXTRAER FEATURES
# ============================================================

def extract_features_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    features = {}

    for col in BAND_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values

            if len(values) > 0:
                features[f"{col}_mean"] = np.mean(values)
                features[f"{col}_std"] = np.std(values)
                features[f"{col}_min"] = np.min(values)
                features[f"{col}_max"] = np.max(values)
                features[f"{col}_median"] = np.median(values)

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

    if "T7_mu" in df.columns and "T7_beta" in df.columns:
        mu = pd.to_numeric(df["T7_mu"], errors="coerce").mean()
        beta = pd.to_numeric(df["T7_beta"], errors="coerce").mean()
        features["T7_mu_beta_ratio"] = mu / (beta + eps)

    if "T8_mu" in df.columns and "T8_beta" in df.columns:
        mu = pd.to_numeric(df["T8_mu"], errors="coerce").mean()
        beta = pd.to_numeric(df["T8_beta"], errors="coerce").mean()
        features["T8_mu_beta_ratio"] = mu / (beta + eps)

    return features


# ============================================================
# 4. CARGAR DATASET
# ============================================================

def load_dataset(dataset_dir):
    csv_files = [
        f for f in glob.glob(os.path.join(dataset_dir, "*.csv"))
        if os.path.basename(f).lower().startswith("abrir-cerrar_mano_")
    ]

    if len(csv_files) == 0:
        raise FileNotFoundError("No se han encontrado CSV válidos.")

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
        rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# 5. EVALUAR MODELOS
# ============================================================

def evaluate_experiment(dataset, feature_cols, experiment_name):
    print("\n================================================")
    print(f"EXPERIMENTO: {experiment_name}")
    print("================================================")

    X = dataset[feature_cols].values
    y = dataset["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("Número de muestras:", X.shape[0])
    print("Número de features:", X.shape[1])
    print("Clases:", list(le.classes_))

    cv_repeated = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

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
            random_state=42
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y_encoded,
            cv=cv_repeated,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro"
            },
            return_train_score=True,
            n_jobs=-1
        )

        results.append({
            "experiment": experiment_name,
            "model": name,
            "train_acc_mean": scores["train_accuracy"].mean(),
            "train_acc_std": scores["train_accuracy"].std(),
            "test_acc_mean": scores["test_accuracy"].mean(),
            "test_acc_std": scores["test_accuracy"].std(),
            "balanced_acc_mean": scores["test_balanced_accuracy"].mean(),
            "balanced_acc_std": scores["test_balanced_accuracy"].std(),
            "f1_macro_mean": scores["test_f1_macro"].mean(),
            "f1_macro_std": scores["test_f1_macro"].std()
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="balanced_acc_mean", ascending=False)

    print("\nRESULTADOS VALIDACIÓN CRUZADA REPETIDA:")
    print(results_df.to_string(index=False))

    # ========================================================
    # GridSearch SVM RBF
    # ========================================================

    print("\n================================================")
    print("OPTIMIZACIÓN SVM RBF")
    print("================================================")

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

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
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X, y_encoded)

    grid_results = pd.DataFrame(grid.cv_results_)
    grid_results = grid_results.sort_values(by="mean_test_score", ascending=False)

    print("Mejores parámetros:", grid.best_params_)
    print("Mejor balanced accuracy:", round(grid.best_score_, 4))

    print("\nTop 10 SVM RBF:")
    print(grid_results[[
        "param_clf__C",
        "param_clf__gamma",
        "mean_train_score",
        "mean_test_score",
        "std_test_score"
    ]].head(10).to_string(index=False))

    return results_df, grid_results


# ============================================================
# 6. MAIN
# ============================================================

dataset = load_dataset(DATASET_DIR)

print("\n================================================")
print("DATASET CARGADO")
print("================================================")

print("Tamaño:", dataset.shape)
print("\nEtiquetas:")
print(dataset["label"].value_counts())

if "DESCONOCIDO" in dataset["label"].values:
    print("Hay archivos sin etiqueta derecha/izquierda.")
    print(dataset[dataset["label"] == "DESCONOCIDO"]["file"].head(20))
    raise SystemExit

metadata_cols = ["label", "file"]

# ----------------------------
# Experimento 1: todas features
# ----------------------------

feature_cols_all = [
    col for col in dataset.columns
    if col not in metadata_cols
]

results_all, grid_all = evaluate_experiment(
    dataset,
    feature_cols_all,
    "Todas las features"
)

# ----------------------------
# Experimento 2: solo T7/T8
# ----------------------------

selected_features = [
    "T7_theta_mean",
    "T7_mu_mean",
    "T7_beta_mean",
    "T8_theta_mean",
    "T8_mu_mean",
    "T8_beta_mean",

    "T7_theta_std",
    "T7_mu_std",
    "T7_beta_std",
    "T8_theta_std",
    "T8_mu_std",
    "T8_beta_std",

    "theta_diff_T7_T8",
    "mu_diff_T7_T8",
    "beta_diff_T7_T8",

    "theta_ratio_T7_T8",
    "mu_ratio_T7_T8",
    "beta_ratio_T7_T8",

    "theta_asym_T7_T8",
    "mu_asym_T7_T8",
    "beta_asym_T7_T8",

    "T7_mu_beta_ratio",
    "T8_mu_beta_ratio"
]

feature_cols_t7t8 = [
    f for f in selected_features
    if f in dataset.columns
]

results_t7t8, grid_t7t8 = evaluate_experiment(
    dataset,
    feature_cols_t7t8,
    "Solo T7/T8 + lateralidad"
)

# ============================================================
# 7. GUARDAR RESULTADOS
# ============================================================

dataset.to_csv(
    os.path.join(DATASET_DIR, "dataset_features_extraidas_final.csv"),
    index=False
)

results_all.to_csv(
    os.path.join(DATASET_DIR, "resultados_repeated_cv_todas_features.csv"),
    index=False
)

grid_all.to_csv(
    os.path.join(DATASET_DIR, "grid_svm_todas_features_final.csv"),
    index=False
)

results_t7t8.to_csv(
    os.path.join(DATASET_DIR, "resultados_repeated_cv_t7t8.csv"),
    index=False
)

grid_t7t8.to_csv(
    os.path.join(DATASET_DIR, "grid_svm_t7t8_final.csv"),
    index=False
)

print("\n================================================")
print("FIN DEL ANÁLISIS")
print("================================================")
