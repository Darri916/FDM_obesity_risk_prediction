"""
MODEL TRAINING
Train multiple ML models and select the best one
Different algorithms work better for different problems
Output: Best trained model + performance visualizations
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MODEL TRAINING")
print("="*70)

# LOAD PROCESSED DATA
print("\n[1/5] Loading Processed Data...")
X_train = pd.read_csv('../data/processed/X_train.csv')
X_val = pd.read_csv('../data/processed/X_val.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
y_val = pd.read_csv('../data/processed/y_val.csv').values.ravel()

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Validation set: {X_val.shape}")
print(f"✓ Number of classes: {len(np.unique(y_train))}")

# Load preprocessor to get class names
with open('../models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
    class_names = preprocessor['target_encoder'].classes_

print(f"✓ Class names: {list(class_names)}")

# INITIALIZE MODELS
print("\n[2/5] Initializing ML Models...")
models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        random_state=42
    ),
    
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),
    
    'XGBoost': XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss'
    )
}

print(f"✓ Initialized {len(models)} models")

# TRAIN AND EVALUATE MODELS
print("\n[3/5] Training Models (this may take 5-10 minutes)...")
results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
    f1_macro = f1_score(y_val, y_val_pred, average='macro')
    
    # Store results
    results[model_name] = {
        'model': model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'predictions': y_val_pred
    }
    
    print(f"    Train Accuracy: {train_acc:.4f}")
    print(f"    Val Accuracy:   {val_acc:.4f}")
    print(f"    F1-Score (Wgt): {f1_weighted:.4f}")
    print(f"    ✓ Completed")

# FIND BEST MODEL
print("\n[4/5] Selecting Best Model...")
best_model_name = max(results, key=lambda x: results[x]['f1_weighted'])
best_model = results[best_model_name]['model']
best_f1 = results[best_model_name]['f1_weighted']

print(f"\n   BEST MODEL: {best_model_name}")
print(f"  F1-Score: {best_f1:.4f}")
print(f"  Accuracy: {results[best_model_name]['val_accuracy']:.4f}")

# CREATE VISUALIZATIONS
print("\n[5/5] Creating Visualizations...")

# Plot 1: Model Comparison
metrics = ['train_accuracy', 'val_accuracy', 'f1_weighted', 'f1_macro']
model_names_list = list(results.keys())
data = {metric: [results[name][metric] for name in model_names_list] for metric in metrics}
df_results = pd.DataFrame(data, index=model_names_list)

plt.figure(figsize=(14, 8))
df_results.plot(kind='bar', width=0.8, colormap='viridis')
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend(title='Metrics', loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../models/model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved model_comparison.png")
plt.close()

# Plot 2: Confusion Matrix for Best Model
y_val_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_val, y_val_pred_best)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           linewidths=0.5, linecolor='gray')
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Class', fontsize=13)
plt.xlabel('Predicted Class', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved confusion_matrix.png")
plt.close()

# Plot 3: Per-Class Performance
report_dict = classification_report(y_val, y_val_pred_best, target_names=class_names, output_dict=True)
classes = list(class_names)
precision = [report_dict[c]['precision'] for c in classes]
recall = [report_dict[c]['recall'] for c in classes]
f1 = [report_dict[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
ax.bar(x - width, precision, width, label='Precision', color='skyblue')
ax.bar(x, recall, width, label='Recall', color='lightcoral')
ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

ax.set_xlabel('Obesity Classes', fontsize=13)
ax.set_ylabel('Score', fontsize=13)
ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('../models/class_performance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved class_performance.png")
plt.close()

# PRINT DETAILED REPORT
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_val, y_val_pred_best, target_names=class_names))

# SAVE BEST MODEL
with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Best model saved to models/best_model.pkl")

# SAVE RESULTS
results_to_save = {
    name: {k: v for k, v in res.items() if k != 'model' and k != 'predictions'}
    for name, res in results.items()
}

with open('../models/training_results.pkl', 'wb') as f:
    pickle.dump({
        'results': results_to_save,
        'best_model_name': best_model_name
    }, f)
print("✓ Training results saved to models/training_results.pkl")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nFinal Model: {best_model_name}")
print(f"Validation Accuracy: {results[best_model_name]['val_accuracy']:.2%}")
print(f"F1-Score (Weighted): {best_f1:.4f}")
print("\nGenerated Files:")
print("  ✓ best_model.pkl")
print("  ✓ model_comparison.png")
print("  ✓ confusion_matrix.png")
print("  ✓ class_performance.png")
print("="*70)