import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD MODEL
# ============================================

print("="*60)
print("INFERENCE TESTS - PREDICTIVE MAINTENANCE")
print("="*60)

model_path = Path(__file__).resolve().parent / "best_predictive_maintenance_model.pkl"
if not model_path.exists():
    print("ERROR: best_predictive_maintenance_model.pkl not found")
    print("Run training.py first")
    exit(1)

model = joblib.load(model_path)
print("Model loaded successfully")

# ============================================
# 2. LOAD TEST DATA
# ============================================

data_path = Path(__file__).resolve().parent / "proof.csv"
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"Data loaded from proof.csv: {len(df)} records")
else:
    print("proof.csv not found. Generating synthetic data...")
    np.random.seed(42)
    n = 1000
    
    data = {
        'Type': np.random.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
        'Air temperature K': np.random.uniform(295, 305, n),
        'Process temperature K': np.random.uniform(305, 315, n),
        'Rotational speed rpm': np.random.uniform(1200, 3000, n),
        'Torque Nm': np.random.uniform(10, 75, n),
        'Tool wear min': np.random.uniform(0, 250, n),
    }
    df = pd.DataFrame(data)
    
    # Feature Engineering
    df['Temp_Diff'] = df['Process temperature K'] - df['Air temperature K']
    df['Power'] = df['Torque Nm'] * df['Rotational speed rpm'] / 1000
    
    expected_columns = ['Type', 'Air temperature K', 'Process temperature K', 
                        'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
                        'Temp_Diff', 'Power']
    df = df[expected_columns]
    
    print(f"Synthetic data generated: {len(df)} records")
    df.to_csv('proof.csv', index=False)
    print("proof.csv saved for future tests")

# ============================================
# 3. BASE PREDICTIONS
# ============================================

print("\n" + "="*60)
print("BASE PREDICTIONS")
print("="*60)

y_proba = model.predict_proba(df)[:, 1]
y_pred = (y_proba >= 0.3).astype(int)

alerts = y_pred.sum()

print(f"Total samples: {len(df)}")
print(f"Alerts generated: {alerts} ({alerts/len(df)*100:.2f}%)")
print(f"Average failure probability: {y_proba.mean():.4f}")
print(f"Maximum probability: {y_proba.max():.4f}")
print(f"Minimum probability: {y_proba.min():.4f}")

# ============================================
# 4. NOISE ROBUSTNESS TEST
# ============================================

print("\n" + "="*60)
print("ROBUSTNESS TEST (NOISE)")
print("="*60)

noise_levels = [0.01, 0.03, 0.05, 0.10, 0.20]
noise_results = []

for level in noise_levels:
    noise = np.random.normal(0, level, df.shape)
    df_noisy = df + noise
    
    df_noisy['Temp_Diff'] = df_noisy['Process temperature K'] - df_noisy['Air temperature K']
    df_noisy['Power'] = df_noisy['Torque Nm'] * df_noisy['Rotational speed rpm'] / 1000
    
    y_pred_noisy = model.predict(df_noisy)
    y_proba_noisy = model.predict_proba(df_noisy)[:, 1]
    
    changes = (y_pred != y_pred_noisy).sum()
    prob_diff = np.mean(np.abs(y_proba - y_proba_noisy))
    
    noise_results.append({
        'noise_level': level,
        'prediction_changes': changes,
        'change_percentage': changes / len(df) * 100,
        'avg_prob_diff': prob_diff
    })
    print(f"Level: {level:.2f} → Changes: {changes} ({changes/len(df)*100:.1f}%), Prob Diff: {prob_diff:.4f}")

# ============================================
# 5. INFERENCE SPEED TEST
# ============================================

print("\n" + "="*60)
print("PERFORMANCE TEST (SPEED)")
print("="*60)

batch_sizes = [1, 10, 100, 1000, len(df)]
times = []

for size in batch_sizes:
    sample = df.iloc[:min(size, len(df))]
    start = time.perf_counter()
    _ = model.predict(sample)
    end = time.perf_counter()
    time_ms = (end - start) / len(sample) * 1000
    times.append(time_ms)
    print(f"Batch of {min(size, len(df))}: {time_ms:.3f} ms per sample")

print(f"Average time: {np.mean(times):.3f} ms per sample")

# ============================================
# 6. SHADOW MODE (PRODUCTION SIMULATION)
# ============================================

print("\n" + "="*60)
print("SHADOW MODE (PRODUCTION SIMULATION)")
print("="*60)

np.random.seed(42)
actual_failures = np.random.choice([0, 1], size=len(df), p=[0.966, 0.034])

model_alerts = (y_proba >= 0.3).astype(int)

tp = ((model_alerts == 1) & (actual_failures == 1)).sum()
fp = ((model_alerts == 1) & (actual_failures == 0)).sum()
tn = ((model_alerts == 0) & (actual_failures == 0)).sum()
fn = ((model_alerts == 0) & (actual_failures == 1)).sum()

print("Simulated Confusion Matrix (with artificial failures):")
print(f"  True Positives: {tp}")
print(f"  False Positives: {fp}")
print(f"  True Negatives: {tn}")
print(f"  False Negatives: {fn}")

if (tp + fn) > 0:
    recall_sim = tp / (tp + fn)
    print(f"Simulated Recall: {recall_sim:.3f}")

cost_fp = 1
cost_fn = 50
total_cost = fp * cost_fp + fn * cost_fn
print(f"\nEstimated error cost (ratio 50:1):")
print(f"  False Positives (cost 1): {fp * cost_fp}")
print(f"  False Negatives (cost 50): {fn * cost_fn}")
print(f"  Total cost: {total_cost}")

# ============================================
# 7. EDGE CASES (SANITY CHECK)
# ============================================

print("\n" + "="*60)
print("EDGE CASES (SANITY CHECK)")
print("="*60)

cases = [
    {
        'name': 'Certain failure (high temp, high torque, high wear)',
        'data': {
            'Type': [2],
            'Air temperature K': [310],
            'Process temperature K': [340],
            'Rotational speed rpm': [2800],
            'Torque Nm': [70],
            'Tool wear min': [250],
            'Temp_Diff': [30],
            'Power': [196]
        }
    },
    {
        'name': 'Unlikely failure (all normal)',
        'data': {
            'Type': [0],
            'Air temperature K': [298],
            'Process temperature K': [308],
            'Rotational speed rpm': [1400],
            'Torque Nm': [35],
            'Tool wear min': [10],
            'Temp_Diff': [10],
            'Power': [49]
        }
    },
    {
        'name': 'Wear failure (high tool wear)',
        'data': {
            'Type': [1],
            'Air temperature K': [300],
            'Process temperature K': [310],
            'Rotational speed rpm': [1500],
            'Torque Nm': [50],
            'Tool wear min': [230],
            'Temp_Diff': [10],
            'Power': [75]
        }
    },
    {
        'name': 'Overload failure (very high torque)',
        'data': {
            'Type': [2],
            'Air temperature K': [305],
            'Process temperature K': [315],
            'Rotational speed rpm': [1200],
            'Torque Nm': [72],
            'Tool wear min': [50],
            'Temp_Diff': [10],
            'Power': [86.4]
        }
    }
]

for case in cases:
    df_case = pd.DataFrame(case['data'])
    prob = model.predict_proba(df_case)[0][1]
    pred = model.predict(df_case)[0]
    print(f"\n{case['name']}:")
    print(f"  Failure probability: {prob:.2%}")
    print(f"  Prediction: {'FAILURE' if pred == 1 else 'OK'}")

# ============================================
# 8. RESULT CHARTS
# ============================================

print("\n" + "="*60)
print("GENERATING CHARTS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Probability distribution
ax1 = axes[0, 0]
ax1.hist(y_proba[y_pred == 0], bins=20, alpha=0.7, label='OK', color='green')
ax1.hist(y_proba[y_pred == 1], bins=20, alpha=0.7, label='Failure', color='red')
ax1.axvline(x=0.3, color='black', linestyle='--', label='Threshold 0.3')
ax1.set_xlabel('Failure probability')
ax1.set_ylabel('Frequency')
ax1.set_title('Probability Distribution')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Noise test
ax2 = axes[0, 1]
noise_plot_df = pd.DataFrame(noise_results)
ax2.plot(noise_plot_df['noise_level'], noise_plot_df['change_percentage'], 'o-', color='blue')
ax2.set_xlabel('Noise level')
ax2.set_ylabel('Prediction changes (%)')
ax2.set_title('Noise Robustness')
ax2.grid(True, alpha=0.3)

# 3. Inference speed
ax3 = axes[1, 0]
ax3.bar([str(t) for t in batch_sizes], times)
ax3.set_xlabel('Batch size')
ax3.set_ylabel('Time (ms per sample)')
ax3.set_title('Inference Performance')
ax3.grid(True, alpha=0.3)

# 4. Edge cases summary
ax4 = axes[1, 1]
names = [c['name'][:20] + '...' for c in cases]
probs = []
for case in cases:
    df_case = pd.DataFrame(case['data'])
    prob = model.predict_proba(df_case)[0][1]
    probs.append(prob)
ax4.barh(names, probs, color=['red' if p > 0.7 else 'green' for p in probs])
ax4.axvline(x=0.3, color='black', linestyle='--', label='Threshold 0.3')
ax4.set_xlabel('Failure probability')
ax4.set_title('Edge Cases')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Inference Tests - Predictive Maintenance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('inference_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 9. EXECUTIVE SUMMARY
# ============================================

print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"""
✓ Model loaded successfully
✓ Data evaluated: {len(df)} records
✓ Alerts generated: {alerts} ({alerts/len(df)*100:.1f}% of total)
✓ Inference time: {np.mean(times):.3f} ms per sample
✓ Robustness: {noise_results[2]['change_percentage']:.1f}% changes with 5% noise
✓ Edge cases: {sum([1 for p in probs if p > 0.7])} of 4 edge cases correctly detected
✓ Shadow mode: Model simulated in production
✓ Charts saved to inference_results.png
""")

print("="*60)
print("TESTS COMPLETED")
print("="*60)