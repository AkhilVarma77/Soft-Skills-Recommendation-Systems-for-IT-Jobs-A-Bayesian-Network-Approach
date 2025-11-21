import os, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn.metrics import f1_score
from sklearn.preprocessing import KBinsDiscretizer

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SME_PATH = os.path.join(OUTPUT_DIR, "sme_dataset.csv")
ADV_PATH = os.path.join(OUTPUT_DIR, "adv_dataset.csv")
TEST_PATH = os.path.join(OUTPUT_DIR, "test_dataset.csv")

LABELS = [
    "communication_interpersonal","teamwork","problemsolving_analyticalthinking",
    "timemanagement","leadership","creativity","stressmanagement","independent_selfmotivated"
]

def bucket_exp_numeric(years):
    if years < 3.5: return "y1_below_3"
    elif years <= 7.5: return "y2_3_7"
    else: return "y3_7_up"

print("Loading datasets...")
sme_df = pd.read_csv(SME_PATH)
adv_df = pd.read_csv(ADV_PATH)
test_df = pd.read_csv(TEST_PATH)

for df in (sme_df, adv_df, test_df):
    df["exp_level"] = df["work_experience_years"].apply(bucket_exp_numeric)



# --- Build & train Msme on SME data (expert CPTs learned from SME samples) ---
print("Training Msme (trained on SME samples)...")

msme_structure = [("job_title", s) for s in LABELS] + [("exp_level", s) for s in LABELS]
msme = BayesianNetwork(msme_structure)

msme.fit(sme_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
msme_infer = VariableElimination(msme)
pickle.dump(msme, open(os.path.join(OUTPUT_DIR, "msme_model.pkl"), "wb"))

# --- Build & train Madv on Advertisement data (ads-based samples) ---
print("Training Madv (trained on Advertisement samples)...")
madv_structure = [("job_title", s) for s in LABELS] + [("exp_level", s) for s in LABELS]
madv = BayesianNetwork(madv_structure)

madv.fit(adv_df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=6)
madv_infer = VariableElimination(madv)
pickle.dump(madv, open(os.path.join(OUTPUT_DIR, "madv_model.pkl"), "wb"))

# --- Evaluation utilities ---
def predict_probs(infer, row):
    ev = {"job_title": row["job_title"], "exp_level": row["exp_level"]}
    probs = {}
    for s in LABELS:
        try:
            q = infer.query([s], evidence=ev)
            probs[s] = float(q.values[1])
        except Exception:
            probs[s] = 0.5
    return probs

def evaluate_model(infer, df, threshold=0.5):
    y_true = []
    y_pred = []
    for _, r in df.iterrows():
        probs = predict_probs(infer, r)
        y_true.append([int(r[s]) for s in LABELS])
        y_pred.append([1 if probs[s] > threshold else 0 for s in LABELS])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0)
    return acc, f1

# Use threshold 0.6 to make predictions slightly stricter (paper-like)
THRESH = 0.6

print("Evaluating on the mixed test set...")
msme_acc, msme_f1 = evaluate_model(msme_infer, test_df, threshold=THRESH)
madv_acc, madv_f1 = evaluate_model(madv_infer, test_df, threshold=THRESH)

print("\nOverall Results on mixed test set:")
print(f"Msme → Accuracy={msme_acc:.4f} | F1={msme_f1:.4f}")
print(f"Madv → Accuracy={madv_acc:.4f} | F1={madv_f1:.4f}")

# Save global bar chart
plt.figure(figsize=(6,4))
models = ["Msme", "Madv"]
accs = [msme_acc*100, madv_acc*100]
plt.bar(models, accs, color=["#1e90ff","#ff4d4d"])
plt.ylim(0,100); plt.ylabel("Accuracy (%)")
plt.title("Bayesian Models Accuracy Comparison")
for i,v in enumerate(accs): plt.text(i, v+1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bayesian_accuracy.png"))
plt.show()

# --- Accuracy by Job Title (paper figure style) ---
print("Plotting accuracy by job title...")
job_titles = sorted(test_df["job_title"].unique())
msme_job_acc = []
madv_job_acc = []
job_labels = []
for jt in job_titles:
    subset = test_df[test_df["job_title"] == jt]
    if len(subset) < 10:
        # skip too-small groups (keeps plot stable)
        continue
    a_msme, _ = evaluate_model(msme_infer, subset, threshold=THRESH)
    a_madv, _ = evaluate_model(madv_infer, subset, threshold=THRESH)
    job_labels.append(jt)
    msme_job_acc.append(a_msme * 100)
    madv_job_acc.append(a_madv * 100)

plt.figure(figsize=(12,5))
plt.plot(job_labels, msme_job_acc, marker='o', color='royalblue', label="Msme (%)")
plt.plot(job_labels, madv_job_acc, marker='s', color='red', label="Madv (%)")
plt.xticks(rotation=60, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of Msme and Madv by Job Title")
plt.legend()
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_by_job_title.png"))
plt.show()

# --- Accuracy by Soft Skill (paper figure style) ---
print("Plotting accuracy by soft skills...")
msme_skill_acc = []
madv_skill_acc = []
skill_labels = [s.replace("_", " ").title() for s in LABELS]

for skill in LABELS:
    # Msme
    y_true = []
    y_pred = []
    for _, r in test_df.iterrows():
        probs = predict_probs(msme_infer, r)
        y_true.append(int(r[skill]))
        y_pred.append(1 if probs[skill] > THRESH else 0)
    msme_skill_acc.append(np.mean(np.array(y_true) == np.array(y_pred)) * 100)

    # Madv
    y_true = []
    y_pred = []
    for _, r in test_df.iterrows():
        probs = predict_probs(madv_infer, r)
        y_true.append(int(r[skill]))
        y_pred.append(1 if probs[skill] > THRESH else 0)
    madv_skill_acc.append(np.mean(np.array(y_true) == np.array(y_pred)) * 100)

plt.figure(figsize=(10,4))
plt.plot(skill_labels, msme_skill_acc, marker='o', color='royalblue', label="Msme (%)")
plt.plot(skill_labels, madv_skill_acc, marker='s', color='red', label="Madv (%)")
plt.xticks(rotation=35, ha='right')
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of Msme and Madv by Soft Skills")
plt.legend()
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_by_softskills.png"))
plt.show()

print("\nSaved charts in 'models/' directory:")
print(" - bayesian_accuracy.png")
print(" - accuracy_by_job_title.png")
print(" - accuracy_by_softskills.png")
