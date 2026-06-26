import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN
# =========================================================
OUTPUT_DIR    = "experimento_3"
FS            = 128
CHANNEL_NAMES = ["AF3", "T7", "Pz", "T8", "AF4"]
BAND_COLS = [
    "AF3_theta", "AF3_alpha", "AF3_beta",
    "T7_theta",  "T7_mu",     "T7_beta",
    "Pz_alpha",  "Pz_beta",
    "T8_theta",  "T8_mu",     "T8_beta",
    "AF4_theta", "AF4_alpha", "AF4_beta",
]

# =========================================================
# EXTRACCIÓN DE FEATURES
# =========================================================
def extract_features(df):
    features = {}

    # Features espectrales (bandas)
    for col in BAND_COLS:
        if col in df.columns:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"]  = df[col].std()

    for band in ["theta", "alpha", "beta"]:
        af3_col = f"AF3_{band}"
        af4_col = f"AF4_{band}"
        if af3_col in df.columns and af4_col in df.columns:
            features[f"AF3_AF4_{band}_diff"]  = df[af3_col].mean() - df[af4_col].mean()
            features[f"AF3_AF4_{band}_ratio"] = df[af3_col].mean() / (df[af4_col].mean() + 1e-10)

    for band in ["theta", "mu", "beta"]:
        t7_col = f"T7_{band}"
        t8_col = f"T8_{band}"
        if t7_col in df.columns and t8_col in df.columns:
            features[f"T7_T8_{band}_diff"] = df[t7_col].mean() - df[t8_col].mean()

    if "AF3_alpha" in df.columns and "AF4_alpha" in df.columns:
        af3_a = df["AF3_alpha"].mean()
        af4_a = df["AF4_alpha"].mean()
        features["frontal_alpha_asymmetry"] = (af4_a - af3_a) / (af4_a + af3_a + 1e-10)

    if "AF3_theta" in df.columns and "AF3_beta" in df.columns:
        features["AF3_theta_beta_ratio"] = df["AF3_theta"].mean() / (df["AF3_beta"].mean() + 1e-10)
    if "AF4_theta" in df.columns and "AF4_beta" in df.columns:
        features["AF4_theta_beta_ratio"] = df["AF4_theta"].mean() / (df["AF4_beta"].mean() + 1e-10)

    # Features de amplitud de señal cruda (AF3 y AF4)
    for ch in ["AF3", "AF4"]:
        if ch in df.columns:
            signal = df[ch].values.astype(float)
            signal = signal - np.mean(signal)

            features[f"{ch}_raw_max"]       = np.max(signal)
            features[f"{ch}_raw_min"]       = np.min(signal)
            features[f"{ch}_raw_range"]     = np.max(signal) - np.min(signal)
            features[f"{ch}_raw_std"]       = np.std(signal)
            features[f"{ch}_raw_var"]       = np.var(signal)
            features[f"{ch}_raw_abs_mean"]  = np.mean(np.abs(signal))

            threshold = np.percentile(np.abs(signal), 75)
            peaks, _ = find_peaks(np.abs(signal), height=threshold, distance=10)
            features[f"{ch}_n_peaks"]       = len(peaks)
            features[f"{ch}_peak_mean_amp"] = np.mean(np.abs(signal[peaks])) if len(peaks) > 0 else 0.0

    return features

def load_dataset(data_dir):
    records = []
    for filepath in glob.glob(os.path.join(data_dir, "*.csv")):
        filename = os.path.basename(filepath)
        parts = filename.rsplit(".", 2)
        if len(parts) < 3:
            continue
        label = parts[0]
        df = pd.read_csv(filepath)
        features = extract_features(df)
        features["label"] = label
        features["file"]  = filename
        records.append(features)
    return pd.DataFrame(records)

# =========================================================
# MAIN
# =========================================================
print("=" * 60)
print("CARGANDO DATASET — EXPERIMENTO 3")
print("=" * 60)

df = load_dataset(OUTPUT_DIR)
print(f"\nMuestras por clase:")
print(df["label"].value_counts().to_string())

df = df[df["label"].isin(["pelota_botando", "pestañear", "reposo"])]
print(f"\nTras cargar las tres clases: {len(df)} muestras")

le = LabelEncoder()
y = le.fit_transform(df["label"])
print(f"\nClases: {dict(enumerate(le.classes_))}")

feature_cols = [c for c in df.columns if c not in ["label", "file"]]
X = df[feature_cols].fillna(0)
print(f"Features: {len(feature_cols)}")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
param_grid = {
    "clf__C":     [0.1, 1, 2, 5, 10, 20, 50],
    "clf__gamma": ["scale", "auto", 0.001, 0.005, 0.01, 0.05],
}

# =========================================================
# VALIDACIÓN CRUZADA
# =========================================================
print("\n" + "=" * 60)
print("VALIDACIÓN CRUZADA")
print("=" * 60)

models = {
    "Dummy baseline":      DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM lineal":          Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", probability=True))]),
    "SVM RBF":             Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf",    probability=True))]),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
    results.append({"model": name, "balanced_acc_mean": scores.mean(), "balanced_acc_std": scores.std()})

results_df = pd.DataFrame(results).sort_values("balanced_acc_mean", ascending=False)
print("\nResultados ordenados por balanced accuracy:")
print(results_df.to_string(index=False))

# =========================================================
# GRIDSEARCH SVM RBF
# =========================================================
print("\n" + "=" * 60)
print("OPTIMIZACIÓN SVM RBF CON GRIDSEARCH")
print("=" * 60)

pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))])
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1, return_train_score=True)
grid.fit(X, y)

print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor balanced accuracy: {grid.best_score_:.4f}")

top10 = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False).head(10)
print("\nTop 10 SVM RBF:")
print(top10[["param_clf__C", "param_clf__gamma", "mean_train_score", "mean_test_score", "std_test_score"]].to_string(index=False))

# =========================================================
# TRAIN/TEST MANUAL CON MEJOR SVM
# =========================================================
print("\n" + "=" * 60)
print("TRAIN/TEST MANUAL CON MEJOR SVM")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy test:          {acc:.4f}")
print(f"Balanced accuracy test: {acc:.4f}")
print(f"\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================================================
# GUARDAR MODELO — entrenado sobre todos los datos
# =========================================================
print("\n" + "=" * 60)
print("GUARDANDO MODELO")
print("=" * 60)

final_model = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True,
                         C=grid.best_params_["clf__C"], gamma=grid.best_params_["clf__gamma"]))])
final_model.fit(X, y)

model_data = {
    "model":         final_model,
    "label_encoder": le,
    "feature_cols":  feature_cols,
}

pkl_path = os.path.join(OUTPUT_DIR, "modelo_svm_experimento3.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"Modelo guardado en: {pkl_path}")
print(f"Parámetros finales: C={grid.best_params_['clf__C']}, gamma={grid.best_params_['clf__gamma']}")
print(f"Features: {len(feature_cols)}")
print("\n" + "=" * 60)
print("FIN DEL ANÁLISIS")
print("=" * 60)