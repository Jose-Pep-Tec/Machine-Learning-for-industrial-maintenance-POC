import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DATA LOADING AND CLEANING
# ============================================

file_path = Path(__file__).resolve().parent / "ai4i2020.csv"
df = pd.read_csv(file_path)

# Clean column names for XGBoost compatibility
df.columns = df.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '').str.strip()

df = df.drop(["UDI", "Product ID"], axis=1)

# Feature Engineering
df['Temp_Diff'] = df['Process temperature K'] - df['Air temperature K']
df['Power'] = df['Torque Nm'] * df['Rotational speed rpm'] / 1000

# Map Type to numeric values
type_mapping = {'L': 0, 'M': 1, 'H': 2}
df['Type'] = df['Type'].map(type_mapping)

# Split features and target
X = df.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df['Machine failure']

# ============================================
# TRAIN + VALIDATION + TEST SPLIT
# ============================================

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ============================================
# MODEL DEFINITION AND GRID SEARCH CONFIG
# ============================================

models = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'param_grid': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'scale_pos_weight': [5, 10, 20]
        }
    },
    'HistGradientBoosting': {
        'model': HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
        'param_grid': {
            'max_iter': [50, 100, 150],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'min_samples_leaf': [10, 20, 30]
        }
    }
}

# ============================================
# GRID SEARCH AND EVALUATION
# ============================================

scorer = make_scorer(recall_score)
results = []

for name, config in models.items():
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print('='*60)
    
    grid = GridSearchCV(
        config['model'],
        config['param_grid'],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scorer,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"\nBest parameters: {grid.best_params_}")
    
    # Evaluate on validation and test sets
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    
    recall_val = recall_score(y_val, y_pred_val)
    recall_test = recall_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_proba_test)
    
    # Store results
    results.append({
        'model': name,
        'best_params': grid.best_params_,
        'recall_val': recall_val,
        'recall_test': recall_test,
        'precision_test': precision_test,
        'f1_test': f1_test,
        'auc_test': auc_test,
        'best_model': best_model
    })
    
    print(f"\nTest Results:")
    print(f"  Recall: {recall_test:.3f}")
    print(f"  Precision: {precision_test:.3f}")
    print(f"  F1: {f1_test:.3f}")
    print(f"  AUC: {auc_test:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

# ============================================
# MODEL COMPARISON
# ============================================

df_results = pd.DataFrame(results)
df_results = df_results.drop('best_model', axis=1)
df_results.to_csv('model_comparison.csv', index=False)

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(df_results[['model', 'recall_test', 'precision_test', 'f1_test', 'auc_test']])

# ============================================
# COMPARISON CHARTS
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = ['recall_test', 'precision_test', 'f1_test', 'auc_test']
titles = ['Test Recall', 'Test Precision', 'F1-Score', 'AUC-ROC']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i // 2, i % 2]
    df_plot = df_results.sort_values(metric, ascending=False)
    colors = ['#2ecc71' if x == df_plot[metric].max() else '#e74c3c' if x == df_plot[metric].min() else '#3498db' for x in df_plot[metric]]
    
    bars = ax.barh(df_plot['model'], df_plot[metric], color=colors)
    ax.axvline(x=df_plot[metric].mean(), color='gray', linestyle='--', alpha=0.7, label='Average')
    ax.set_xlim(0, 1)
    ax.set_xlabel(title)
    ax.set_title(title)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=9)

plt.suptitle('Model Comparison - Predictive Maintenance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# BEST MODEL AND FEATURE IMPORTANCE
# ============================================

best = results[np.argmax([r['recall_test'] for r in results])]
print("\n" + "="*60)
print(f"BEST MODEL: {best['model']} with Recall: {best['recall_test']:.3f}")
print("="*60)

final_model = best['best_model']

if hasattr(final_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importances)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importances, x='importance', y='feature')
    plt.title(f'Feature Importances - {best["model"]}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=300)
    plt.show()
else:
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42)
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)
    print(importances)

# ============================================
# CONFUSION MATRIX OF BEST MODEL
# ============================================

y_pred_final = final_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best["model"]}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

# ============================================
# SAVE FINAL MODEL
# ============================================

import joblib
joblib.dump(final_model, 'best_predictive_maintenance_model.pkl')

print("\n" + "="*60)
print("PROCESS COMPLETED")
print("="*60)
print("Generated files:")
print("  - model_comparison.csv")
print("  - model_comparison.png")
print("  - feature_importances.png")
print("  - confusion_matrix.png")
print("  - best_predictive_maintenance_model.pkl")