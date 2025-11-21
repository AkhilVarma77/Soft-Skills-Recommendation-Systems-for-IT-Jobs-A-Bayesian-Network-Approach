
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

print("✅ Starting Deep Learning Model Training & Plotting")

DATA_PATH = r"models/test_dataset.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    "communication_interpersonal","teamwork","problemsolving_analyticalthinking",
    "timemanagement","leadership","creativity","stressmanagement","independent_selfmotivated"
]

df = pd.read_csv(DATA_PATH)
print(f"📂 Loaded dataset with {len(df)} samples")

# Encode categorical
le_job = LabelEncoder()
le_qual = LabelEncoder()
df["job_title_enc"] = le_job.fit_transform(df["job_title"])
df["qualification_enc"] = le_qual.fit_transform(df["qualification"])

X = df[["job_title_enc","qualification_enc","work_experience_years"]].values
y = df[LABELS].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- TRAIN ----------
mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), activation='relu',
                    learning_rate_init=0.001, max_iter=300, random_state=42)

y_pred_all, y_true_all = [], []
for i, skill in enumerate(LABELS):
    mlp.fit(X_train, y_train[:, i])
    y_pred = mlp.predict(X_test)
    y_pred_all.append(y_pred)
    y_true_all.append(y_test[:, i])

y_pred_all = np.array(y_pred_all).T
y_true_all = np.array(y_true_all).T
global_acc = (y_pred_all == y_true_all).mean()
global_f1 = f1_score(y_true_all.flatten(), y_pred_all.flatten(), average="micro")
print(f"\n📊 DL Results: Accuracy={global_acc:.4f}, F1={global_f1:.4f}")

# ---------- PLOTS ----------
# Accuracy by Job Title
print("📈 Plotting DL Accuracy by Job Title...")
job_titles = sorted(df["job_title"].unique())
dl_job_acc = []
for jt in job_titles:
    subset = df[df["job_title"] == jt]
    if len(subset) < 10: continue
    X_sub = scaler.transform(subset[["job_title_enc","qualification_enc","work_experience_years"]].values)
    y_true_sub = subset[LABELS].values
    y_pred_sub = (mlp.predict_proba(X_sub)[:,1] > 0.5).astype(int)
    dl_job_acc.append(np.mean(y_pred_sub == y_true_sub[:,0]) * 100)

plt.figure(figsize=(10,5))
plt.plot(job_titles[:len(dl_job_acc)], dl_job_acc, marker="^", color="green", label="DL (%)")
plt.xticks(rotation=60, ha="right")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of DL by Job Title")
plt.legend()
plt.grid(alpha=0.4, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"dl_accuracy_by_job_title.png"))
plt.show()

# Accuracy by Soft Skills
print("📈 Plotting DL Accuracy by Soft Skills...")
dl_skill_acc = []
for i, skill in enumerate(LABELS):
    y_pred = mlp.predict(X_test)
    y_true = y_test[:, i]
    acc = accuracy_score(y_true, y_pred)
    dl_skill_acc.append(acc*100)

skill_labels = [s.replace("_"," ").title() for s in LABELS]
plt.figure(figsize=(9,4))
plt.plot(skill_labels, dl_skill_acc, marker="^", color="green", label="DL (%)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of DL by Soft Skills")
plt.legend()
plt.grid(alpha=0.4, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"dl_accuracy_by_softskills.png"))
plt.show()

# ---------- SAVE ----------
dump({
    "model": mlp,
    "scaler": scaler,
    "le_job": le_job,
    "le_qual": le_qual,
    "global_acc": global_acc,
    "global_f1": global_f1
}, os.path.join(OUTPUT_DIR, "dl_model.pkl"))
print("✅ DL model & charts saved successfully!")
