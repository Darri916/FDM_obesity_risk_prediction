"""
DATA PREPROCESSING 
Transform raw data into ML-ready format
ML models need numerical, scaled, clean data
Output: Processed train/val sets + saved preprocessor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STEP 6: DATA PREPROCESSING (FIXED)")
print("="*70)

# Create directories
os.makedirs('../data/processed', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# LOAD DATA
print("\n[1/7] Loading train.csv...")
df = pd.read_csv('../data/train.csv')
print(f"✓ Loaded: {df.shape}")

# REMOVE ID COLUMN
print("\n[2/7] Removing ID column...")
if 'id' in df.columns:
    df = df.drop('id', axis=1)
print(f"✓ Shape after removing ID: {df.shape}")

# SEPARATE TARGET FIRST
print("\n[3/7] Separating target variable...")
target = df['NObeyesdad'].copy()
df_features = df.drop('NObeyesdad', axis=1)
print(f"✓ Features: {df_features.shape}")
print(f"✓ Target: {target.shape}")

# FEATURE ENGINEERING
print("\n[4/7] Creating New Features...")

# BMI (Body Mass Index) - Most important indicator
df_features['BMI'] = df_features['Weight'] / (df_features['Height'] ** 2)
print("  ✓ Created BMI feature")

# Age Groups - will be encoded later
df_features['AgeGroup'] = pd.cut(df_features['Age'], bins=[0, 18, 30, 45, 60, 100],
                        labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior'])
print("  ✓ Created AgeGroup feature")

# Activity Level based on physical activity frequency
df_features['ActivityLevel'] = pd.cut(df_features['FAF'], bins=[0, 1, 2, 3, 4],
                             labels=['Sedentary', 'Light', 'Moderate', 'Active'])
print("  ✓ Created ActivityLevel feature")

# Healthy Eating Score
df_features['HealthyEatingScore'] = df_features['FCVC'] + df_features['NCP'] + (df_features['CH2O'] / 3)
print("  ✓ Created HealthyEatingScore")

# High Technology Use
df_features['HighTechUse'] = (df_features['TUE'] > 2).astype(int)
print("  ✓ Created HighTechUse")

# Active Transportation
df_features['ActiveTransport'] = df_features['MTRANS'].isin(['Walking', 'Bike']).astype(int)
print("  ✓ Created ActiveTransport")

# Calorie to Vegetable Ratio
df_features['CalorieVegetableRatio'] = df_features['FAVC'].map({'yes': 1, 'no': 0}) / (df_features['FCVC'] + 0.1)
print("  ✓ Created CalorieVegetableRatio")

# Water Intake Category
df_features['WaterIntake'] = pd.cut(df_features['CH2O'], bins=[0, 1, 2, 3],
                           labels=['Low', 'Medium', 'High'])
print("  ✓ Created WaterIntake")

# Unhealthy Habits Score
df_features['UnhealthyHabits'] = (df_features['CALC'] != 'no').astype(int) + df_features['SMOKE'].map({'yes': 1, 'no': 0})
print("  ✓ Created UnhealthyHabits")

print(f"\n✓ Total features after engineering: {df_features.shape[1]}")

# ENCODE CATEGORICAL VARIABLES
print("\n[5/7] Encoding Categorical Variables...")

# Get categorical columns (including newly created ones)
categorical_cols = df_features.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"  Found {len(categorical_cols)} categorical columns to encode")

# Initialize encoders dictionary
label_encoders = {}

# Encode each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}")

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(target)
print(f"\n  ✓ Encoded target: {len(target_encoder.classes_)} classes")
print(f"  Classes: {list(target_encoder.classes_)}")

# VERIFY ALL COLUMNS ARE NUMERIC
print("\n[6/7] Verifying data types...")
non_numeric = df_features.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
if len(non_numeric) > 0:
    print(f"  ⚠ WARNING: Non-numeric columns found: {non_numeric}")
    print("  Converting to numeric...")
    for col in non_numeric:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        df_features[col] = df_features[col].fillna(0)
else:
    print("  ✓ All columns are numeric")

# SCALE NUMERICAL FEATURES
print("\n[7/7] Scaling Features...")
scaler = StandardScaler()
numerical_cols = df_features.columns.tolist()

df_features[numerical_cols] = scaler.fit_transform(df_features[numerical_cols])
print(f"  ✓ Scaled {len(numerical_cols)} features")

# SPLIT DATA
print("\nSplitting into Train/Validation Sets...")
X_train, X_val, y_train, y_val = train_test_split(
    df_features, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"  ✓ Training set: {X_train.shape}")
print(f"  ✓ Validation set: {X_val.shape}")

# Show class distribution
print("\n  Class distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for cls_idx, count in zip(unique, counts):
    cls_name = target_encoder.classes_[cls_idx]
    pct = (count / len(y_train)) * 100
    print(f"    {cls_name:30s}: {count:5d} ({pct:5.2f}%)")

# FINAL VERIFICATION
print("\n" + "="*70)
print("FINAL DATA VERIFICATION")
print("="*70)
print(f"X_train dtype check:")
print(X_train.dtypes.value_counts())
print(f"\nAny non-numeric values: {X_train.select_dtypes(exclude=['int64', 'float64']).shape[1]} columns")
print(f"Any NaN values: {X_train.isnull().sum().sum()}")
print(f"Any infinite values: {np.isinf(X_train.values).sum()}")

# SAVE PROCESSED DATA
print("\nSaving Processed Data...")
X_train.to_csv('../data/processed/X_train.csv', index=False)
X_val.to_csv('../data/processed/X_val.csv', index=False)
pd.DataFrame(y_train, columns=['NObeyesdad']).to_csv('../data/processed/y_train.csv', index=False)
pd.DataFrame(y_val, columns=['NObeyesdad']).to_csv('../data/processed/y_val.csv', index=False)
print("  ✓ Saved to data/processed/")

# SAVE PREPROCESSOR
preprocessor_data = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'target_encoder': target_encoder,
    'feature_names': df_features.columns.tolist()
}

with open('../models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor_data, f)
print("  ✓ Saved preprocessor to models/preprocessor.pkl")

print("\n" + "="*70)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\nFinal Dataset Summary:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_val):,}")
print(f"  Features: {X_train.shape[1]}")
print(f"  Classes: {len(target_encoder.classes_)}")
print(f"  All data is numeric: ✓")
print(f"  No missing values: ✓")
print("="*70)