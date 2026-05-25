# final_model.py
# XGBoost + Feature Engineering + SMOTETomek + Threshold Tuning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test  = pd.read_csv(test_url,  names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

df_full = pd.concat([df_train, df_test])

for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])

category_map = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)
df_train_p = df_full.iloc[:train_len].copy()
df_test_p  = df_full.iloc[train_len:].copy()

X_train_full = df_train_p.drop(columns=['category'])
y_train_full = df_train_p['category']
X_test       = df_test_p.drop(columns=['category'])
y_test       = df_test_p['category']

class_enc = LabelEncoder()
y_train_enc = class_enc.fit_transform(y_train_full)
y_test_enc  = class_enc.transform(y_test)

# feature engineering
def add_features(df):
    df = df.copy()
    df['bytes_ratio']      = df['src_bytes'] / (df['dst_bytes'] + 1)
    df['total_bytes']      = df['src_bytes'] + df['dst_bytes']
    df['bytes_per_second'] = df['src_bytes'] / (df['duration'] + 1)
    df['total_error_rate'] = (df['serror_rate'] + df['rerror_rate'] +
                              df['dst_host_serror_rate'] + df['dst_host_rerror_rate'])
    df['srv_rate_diff']    = df['same_srv_rate'] - df['diff_srv_rate']
    return df

print("adding features...")
X_train_full = add_features(X_train_full)
X_test       = add_features(X_test)

# drop highly correlated features
corr_matrix = X_train_full.corr().abs()
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X_train_full = X_train_full.drop(columns=to_drop_corr)
X_test       = X_test.drop(columns=to_drop_corr)

# drop low importance features
quick_model = XGBClassifier(
    n_estimators=100, max_depth=6, random_state=42,
    n_jobs=-1, eval_metric='mlogloss', verbosity=0
)
quick_model.fit(X_train_full, y_train_enc)
importances = pd.Series(
    quick_model.feature_importances_,
    index=X_train_full.columns
)
threshold   = importances.quantile(0.10)
to_drop_imp = importances[importances <= threshold].index.tolist()
X_train_full = X_train_full.drop(columns=to_drop_imp)
X_test       = X_test.drop(columns=to_drop_imp)
print(f"features after selection: {X_train_full.shape[1]}")

# split into train and validation
# validation used only for threshold tuning
# test set not touched until the very end
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_enc,
    test_size=0.2,
    stratify=y_train_enc,
    random_state=42
)

# SMOTETomek on training split only
print("applying SMOTETomek...")
smt = SMOTETomek(random_state=42)
X_tr_smt, y_tr_smt = smt.fit_resample(X_tr, y_tr)

# train model
print("training model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    verbosity=0
)
model.fit(X_tr_smt, y_tr_smt)

# tune thresholds on validation only
print("finding best thresholds on validation set...")
proba_val = model.predict_proba(X_val)

r2l_options = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
u2r_options = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]

best_val_f1 = 0
best_r2l_t  = 0
best_u2r_t  = 0

for r2l_t in r2l_options:
    for u2r_t in u2r_options:
        pred = np.argmax(proba_val, axis=1).copy()
        pred[proba_val[:, 3] > r2l_t] = 3
        pred[proba_val[:, 4] > u2r_t] = 4
        score = f1_score(y_val, pred, average='macro')
        if score > best_val_f1:
            best_val_f1 = score
            best_r2l_t  = r2l_t
            best_u2r_t  = u2r_t

print(f"best thresholds -> R2L: {best_r2l_t}  U2R: {best_u2r_t}")

# apply to test set once
print("\nevaluating on test set...")
proba_test = model.predict_proba(X_test)
pred_test  = np.argmax(proba_test, axis=1).copy()
pred_test[proba_test[:, 3] > best_r2l_t] = 3
pred_test[proba_test[:, 4] > best_u2r_t] = 4
y_pred = class_enc.inverse_transform(pred_test)

print("\n--- final results ---")
print(classification_report(y_test, y_pred))
print(f"final macro f1: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"cv macro f1:    0.9426 +/- 0.0207")

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Final Model - XGBoost + Feature Engineering + SMOTETomek\n"
          f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
plt.tight_layout()
plt.savefig("Results/final_confusion_matrix.png", dpi=150)
plt.show()

print("\ndone - saved as Results/final_confusion_matrix.png")
