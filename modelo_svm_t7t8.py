import os
import glob
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

DATASET_DIR = r"C:\Users\Emilio\Desktop\4_TELECO\TFG_EEG\valores_com_emotiv_128hz"

MODEL_OUTPUT = os.path.join(DATASET_DIR, "modelo_svm_rbf_t7t8.pkl")
FEATURES_OUTPUT = os.path.join(DATASET_DIR, "dataset_features_t7t8.csv")
RESULTS_OUTPUT = os.path.join(DATASET_DIR, "resultados_svm_t7t8.csv")

# Solo bandas de T7 y T8
REQUIRED_COLUMNS = [
    "T7_theta", "T7_mu", "T7_beta",
    "T8_theta", "T8_mu", "T8_beta"
]


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
# 3. EXTRAER FEATURES DE T7 Y T8
# ============================================================

def extract_t7_t8_features(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en {csv_path}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)

    if len(df) == 0:
        raise ValueError(f"No hay datos válidos en {csv_path}")

    eps = 1e-8

    # Valores medios de cada banda
    T7_theta = df["T7_theta"].mean()
    T7_mu = df["T7_mu"].mean()
    T7_beta = df["T7_beta"].mean()

    T8_theta = df["T8_theta"].mean()
    T8_mu = df["T8_mu"].mean()
    T8_beta = df["T8_beta"].mean()

    # Desviaciones típicas
    T7_theta_std = df["T7_theta"].std()
    T7_mu_std = df["T7_mu"].std()
    T7_beta_std = df["T7_beta"].std()

    T8_theta_std = df["T8_theta"].std()
    T8_mu_std = df["T8_mu"].std()
    T8_beta_std = df["T8_beta"].std()

    # Diferencias entre hemisferios/canales
    theta_diff = T7_theta - T8_theta
    mu_diff = T7_mu - T8_mu
    beta_diff = T7_beta - T8_beta

    # Ratios T7/T8
    theta_ratio = T7_theta / (T8_theta + eps)
    mu_ratio = T7_mu / (T8_mu + eps)
    beta_ratio = T7_beta / (T8_beta + eps)

    # Asimetrías normalizadas
    theta_asym = (T7_theta - T8_theta) / (T7_theta + T8_theta + eps)
    mu_asym = (T7_mu - T8_mu) / (T7_mu + T8_mu + eps)
    beta_asym = (T7_beta - T8_beta) / (T7_beta + T8_beta + eps)

    # Relación mu/beta dentro de cada canal
    T7_mu_beta_ratio = T7_mu / (T7_beta + eps)
    T8_mu_beta_ratio = T8_mu / (T8_beta + eps)

    features = {
        "T7_theta_mean": T7_theta,
        "T7_mu_mean": T7_mu,
        "T7_beta_mean": T7_beta,

        "T8_theta_mean": T8_theta,
        "T8_mu_mean": T8_mu,
        "T8_beta_mean": T8_beta,

        "T7_theta_std": T7_theta_std,
        "T7_mu_std": T7_mu_std,
        "T7_beta_std": T7_beta_std,

        "T8_theta_std": T8_theta_std,
        "T8_mu_std": T8_mu_std,
        "T8_beta_std": T8_beta_std,

        "theta_diff_T7_T8": theta_diff,
        "mu_diff_T7_T8": mu_diff,
        "beta_diff_T7_T8": beta_diff,

        "theta_ratio_T7_T8": theta_ratio,
        "mu_ratio_T7_T8": mu_ratio,
        "beta_ratio_T7_T8": beta_ratio,

        "theta_asym_T7_T8": theta_asym,
        "mu_asym_T7_T8": mu_asym,
        "beta_asym_T7_T8": beta_asym,

        "T7_mu_beta_ratio": T7_mu_beta_ratio,
        "T8_mu_beta_ratio": T8_mu_beta_ratio
    }

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

        if label == "DESCONOCIDO":
            print(f"Ignorado por etiqueta desconocida: {filename}")
            continue

        try:
            features = extract_t7_t8_features(csv_path)
            features["label"] = label
            features["file"] = filename
            rows.append(features)

        except Exception as e:
            print(f"Error procesando {filename}: {e}")

    dataset = pd.DataFrame(rows)
    return dataset


# ============================================================
# 5. PROGRAMA PRINCIPAL
# ============================================================

dataset = load_dataset(DATASET_DIR)

print("\n================================================")
print("DATASET CARGADO")
print("================================================")
print("Tamaño:", dataset.shape)
print("\nEtiquetas:")
print(dataset["label"].value_counts())

metadata_cols = ["label", "file"]
feature_cols = [col for col in dataset.columns if col not in metadata_cols]

X = dataset[feature_cols].values
y = dataset["label"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nClases:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"{i}: {clase}")

print("\nNúmero de features usadas:", len(feature_cols))
print("\nFeatures:")
for f in feature_cols:
    print("-", f)


# ============================================================
# 6. COMPARACIÓN RÁPIDA DE MODELOS
# ============================================================

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
    ])
}

results = []

print("\n================================================")
print("VALIDACIÓN CRUZADA REPETIDA")
print("================================================")

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

print(results_df.to_string(index=False))


# ============================================================
# 7. OPTIMIZAR SVM RBF
# ============================================================

print("\n================================================")
print("OPTIMIZACIÓN SVM RBF")
print("================================================")

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

param_grid = {
    "clf__C": [0.1, 0.5, 1, 2, 5, 10, 20, 50],
    "clf__gamma": ["scale", "auto", 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
}

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=inner_cv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    return_train_score=True
)

grid.fit(X, y_encoded)

print("Mejores parámetros:", grid.best_params_)
print("Mejor balanced accuracy:", round(grid.best_score_, 4))

grid_results = pd.DataFrame(grid.cv_results_)
grid_results = grid_results.sort_values(by="mean_test_score", ascending=False)

print("\nTop 10 configuraciones SVM:")
print(grid_results[[
    "param_clf__C",
    "param_clf__gamma",
    "mean_train_score",
    "mean_test_score",
    "std_test_score"
]].head(10).to_string(index=False))


# ============================================================
# 8. ENTRENAR MODELO FINAL CON TODOS LOS DATOS
# ============================================================

best_model = grid.best_estimator_

best_model.fit(X, y_encoded)

model_package = {
    "model": best_model,
    "label_encoder": label_encoder,
    "feature_cols": feature_cols
}

joblib.dump(model_package, MODEL_OUTPUT)

print("\n================================================")
print("MODELO FINAL GUARDADO")
print("================================================")
print(MODEL_OUTPUT)


# ============================================================
# 9. GUARDAR RESULTADOS
# ============================================================

dataset.to_csv(FEATURES_OUTPUT, index=False)
results_df.to_csv(RESULTS_OUTPUT, index=False)

print("\nArchivos guardados:")
print(FEATURES_OUTPUT)
print(RESULTS_OUTPUT)

print("\n================================================")
print("FIN")
print("================================================")
