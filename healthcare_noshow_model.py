import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

print("=" * 80)
print("HEALTHCARE NO-SHOW PREDICTION - XGBoost Model")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading dataset...")

try:
    df = pd.read_csv('https://raw.githubusercontent.com/joniahdotcom/Medical-Appointment-No-Shows/master/medical_appointment_no_show.csv')
    print("✓ Dataset loaded successfully!")
except Exception as e:
    print(f"⚠ Could not load from GitHub: {e}")
    print("Creating sample dataset...")
    np.random.seed(42)
    n_records = 10000
    
    df = pd.DataFrame({
        'Age': np.random.randint(0, 100, n_records),
        'SMS_received': np.random.choice([0, 1], n_records),
        'Alcoholism': np.random.choice([0, 1], n_records),
        'Handicap': np.random.choice([0, 1, 2, 3], n_records),
        'Hypertension': np.random.choice([0, 1], n_records),
        'Diabetes': np.random.choice([0, 1], n_records),
        'Scheduled_day': pd.date_range('2023-01-01', periods=n_records, freq='H'),
        'Appointment_day': pd.date_range('2023-02-01', periods=n_records, freq='H'),
        'No_show': np.random.choice(['No', 'Yes'], n_records, p=[0.79, 0.21])
    })

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[STEP 2] Exploratory Data Analysis...")

print(f"\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nNo-Show Rate:")
if 'No_show' in df.columns:
    no_show_rate = (df['No_show'] == 'Yes').sum() / len(df) * 100
    print(f"  {no_show_rate:.2f}% of patients didn't show up")
    print(f"  {df['No_show'].value_counts()}")

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 3] Data Preprocessing...")

data = df.copy()

# Handle date columns
if 'Scheduled_day' in data.columns and 'Appointment_day' in data.columns:
    data['Scheduled_day'] = pd.to_datetime(data['Scheduled_day'])
    data['Appointment_day'] = pd.to_datetime(data['Appointment_day'])
    data['days_until_appointment'] = (data['Appointment_day'] - data['Scheduled_day']).dt.days
    data['scheduled_day_of_week'] = data['Scheduled_day'].dt.dayofweek
    data['appointment_day_of_week'] = data['Appointment_day'].dt.dayofweek
    data = data.drop(['Scheduled_day', 'Appointment_day'], axis=1)

# Handle target variable
if 'No_show' in data.columns:
    data['No_show'] = (data['No_show'] == 'Yes').astype(int)
    target = 'No_show'
else:
    print("Warning: No_show column not found!")
    target = None

# Remove rows with missing target
if target and data[target].isnull().sum() > 0:
    data = data.dropna(subset=[target])

print(f"Processed dataset shape: {data.shape}")
print(f"Features: {list(data.columns)}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING & MODEL PREPARATION
# ============================================================================
print("\n[STEP 4] Feature Engineering & Preparation...")

X = data.drop(target, axis=1)
y = data[target]

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Class balance:\n{y.value_counts(normalize=True)}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} records")
print(f"Test set: {X_test.shape[0]} records")

# ============================================================================
# STEP 5: TRAIN XGBOOST MODEL
# ============================================================================
print("\n[STEP 5] Training XGBoost Model...")

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale positive weight: {scale_pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    verbose=0
)

model.fit(X_train, y_train)
print("✓ Model training complete!")

# ============================================================================
# STEP 6: EVALUATE MODEL
# ============================================================================
print("\n[STEP 6] Evaluating Model...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = (y_pred == y_test).mean()
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"\n✓ ROC-AUC Score: {roc_auc:.4f}")
print(f"✓ Accuracy: {accuracy * 100:.1f}%")
print(f"✓ Precision: {precision * 100:.1f}%")
print(f"✓ Recall: {recall * 100:.1f}%")
print(f"✓ F1-Score: {f1:.4f}")
print(f"✓ Specificity: {specificity * 100:.1f}%")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Show', 'No-Show']))

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n[STEP 7] Feature Importance Analysis...")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n[STEP 8] Saving Results...")

sample_predictions = pd.DataFrame({
    'Actual': y_test.values[:100],
    'Predicted': y_pred[:100],
    'Probability_NoShow': y_pred_proba[:100]
})

export_data = {
    'timestamp': datetime.now().isoformat(),
    'best_model': 'XGBoost',
    'roc_auc': float(roc_auc),
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'specificity': float(specificity),
    'sample_predictions': sample_predictions.head(50).to_dict(orient='records'),
    'dataset_stats': {
        'total_records': len(df),
        'no_show_rate': float((df[target] == 1).sum() / len(df)) if target else 0,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': X.columns.tolist()
    },
    'top_features': feature_importance.head(10).to_dict(orient='records'),
    'model_config': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'scale_pos_weight': float(scale_pos_weight)
    }
}

with open('model_results.json', 'w') as f:
    json.dump(export_data, f, indent=2, default=str)

print("✓ Results saved to 'model_results.json'")

sample_predictions.to_csv('predictions_sample.csv', index=False)
print("✓ Sample predictions saved to 'predictions_sample.csv'")

feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ Feature importance saved to 'feature_importance.csv'")

# ============================================================================
# STEP 9: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE")
print("=" * 80)

print(f"""
MODEL PERFORMANCE
─────────────────────────────────────
Model: XGBoost
ROC-AUC: {roc_auc:.4f} ({roc_auc * 100:.1f}% accuracy)
Accuracy: {accuracy * 100:.1f}%
Precision: {precision * 100:.1f}%
Recall: {recall * 100:.1f}%
F1-Score: {f1:.4f}
Specificity: {specificity * 100:.1f}%

DATASET STATISTICS
─────────────────────────────────────
Total Records: {len(df):,}
No-Show Rate: {(df[target] == 1).sum() / len(df) * 100:.1f}%
Training Samples: {len(X_train):,}
Test Samples: {len(X_test):,}

TOP 3 FEATURES
─────────────────────────────────────
1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Importance']:.1%}
2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Importance']:.1%}
3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Importance']:.1%}

FILES GENERATED
─────────────────────────────────────
✓ model_results.json (Dashboard data)
✓ predictions_sample.csv (Sample predictions)
✓ feature_importance.csv (Top features)

NEXT STEP
─────────────────────────────────────
Open healthcare_dashboard_professional_v3.html in browser!
""")

print("=" * 80)
