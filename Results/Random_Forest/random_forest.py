import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original class labels
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

# Split back into train and test
train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

os.makedirs("results", exist_ok=True)
labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

os.makedirs("results", exist_ok=True)
labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

def evaluate_model(name, model, run_cv=False):
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    report_text = classification_report(y_test, y_pred, labels=labels)
    print(report_text)

    with open(f"results/{name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write(report_text)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"results/{name}_confusion_matrix.png", dpi=150)
    plt.close()

    cv_mean = None
    cv_std = None

    if run_cv:
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )
        cv_mean = scores.mean()
        cv_std = scores.std()
        print(f"Cross-validation macro F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

        with open(f"results/{name}_cv_score.txt", "w", encoding="utf-8") as f:
            f.write(f"Cross-validation macro F1: {cv_mean:.4f} (+/- {cv_std:.4f})\n")
            f.write(f"Test macro F1: {macro_f1:.4f}\n")

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cv_macro_f1_mean": cv_mean,
        "cv_macro_f1_std": cv_std
    }

results = []

rf_default = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("rf_default", rf_default))


rf_balanced = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("rf_balanced", rf_balanced))


rf_tuned_balanced = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("rf_tuned_balanced", rf_tuned_balanced, run_cv=True))

balanced_rf = BalancedRandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("balanced_random_forest", balanced_rf, run_cv=True))

balanced_rf_500 = BalancedRandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("balanced_random_forest_500", balanced_rf_500, run_cv=True))

balanced_rf_depth20 = BalancedRandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("balanced_random_forest_depth20", balanced_rf_depth20, run_cv=True))

balanced_rf_regularized = BalancedRandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

results.append(evaluate_model("balanced_random_forest_regularized", balanced_rf_regularized))

rf_random_oversampling = Pipeline(steps=[
    ("oversample", RandomOverSampler(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

results.append(evaluate_model("rf_random_oversampling", rf_random_oversampling))

rf_smotenc = Pipeline(steps=[
    ("smote", SMOTENC(
        categorical_features=[1, 2, 3],
        random_state=42,
        k_neighbors=3
    )),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

results.append(evaluate_model("rf_smotenc", rf_smotenc))

summary = pd.DataFrame(results)
summary.to_csv("results/random_forest_summary.csv", index=False)

print("\nSummary:")
print(summary)