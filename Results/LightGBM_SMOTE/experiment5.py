# Network Intrusion Detection - Experiment 5
# LightGBM + SMOTE
# trying LightGBM as an alternative to XGBoost
# so we can compare it against XGBoost with SMOTE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

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

X_train = df_train_p.drop(columns=['category'])
y_train = df_train_p['category']
X_test  = df_test_p.drop(columns=['category'])
y_test  = df_test_p['category']

class_enc = LabelEncoder()
y_train_enc = class_enc.fit_transform(y_train)
y_test_enc  = class_enc.transform(y_test)

print("applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train_enc)

print("class distribution after SMOTE:")
unique, counts = np.unique(y_train_sm, return_counts=True)
for name, cnt in zip(class_enc.classes_, counts):
    print(f"  {name}: {cnt}")

# LightGBM instead of XGBoost
# verbose=-1 suppresses the training output spam
print("\ntraining LightGBM model...")
model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train_sm, y_train_sm)

y_pred_enc = model.predict(X_test)
y_pred = class_enc.inverse_transform(y_pred_enc)

print("\n--- results ---")
print(classification_report(y_test, y_pred))

score = f1_score(y_test, y_pred, average='macro')
print(f"macro f1: {score:.4f}")
print(f"experiment 1 (xgboost baseline):        0.5599")
print(f"experiment 2 (xgboost + smote):         0.6273")
print(f"experiment 3 (xgboost + smotetomek):    0.6485")
print(f"experiment 4 (xgboost + thresholds):    0.6780")
print(f"experiment 5 (lightgbm + smote):        {score:.4f}")

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Experiment 5 - LightGBM + SMOTE\nMacro F1: {score:.4f}")
plt.tight_layout()
plt.savefig("exp5_confusion_matrix.png", dpi=150)
plt.show()

print("\ndone - saved as exp5_confusion_matrix.png")
