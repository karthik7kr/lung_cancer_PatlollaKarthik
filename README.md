import pandas as pd
 import numpy as np
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
 from sklearn.tree import DecisionTreeClassifier
 from sklearn.decomposition import PCA
 from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


 
 # Load the dataset
 data = pd.read_csv('Lung_Cancer_dataset.csv')
 print("Dataset shape:", data.shape)
 print("\nDataset info:")
 print(data.info())
 print("\nMissing values:")
 print(data.isnull().sum())
 Dataset shape: (59, 7)
 Dataset info:
 <class 'pandas.core.frame.DataFrame'>
 RangeIndex: 59 entries, 0 to 58
 Data columns (total 7 columns):
 #   Column   Non-Null Count  Dtype ---  ------   --------------  ----- 
 0   Name     59 non-null     object
 1   Surname  59 non-null     object
 2   Age      59 non-null     int64 
 3   Smokes   59 non-null     int64 
 4   AreaQ    59 non-null     int64 
 5   Alkhol   59 non-null     int64 
 6   Result   59 non-null     int64 
dtypes: int64(5), object(2)
 memory usage: 3.4+ KB
 None
 Missing values:
 Name       0
 Surname    0
 Age        0
 Smokes     0
 AreaQ      0
 Alkhol     0
 Result     0
 dtype: int64




 
 # Removing non-predictive columns and prepare features
 # Drop Name and Surname as they are not predictive features
 X = data.drop(['Name', 'Surname', 'Result'], axis=1)
 y = data['Result']
 print("Features:", X.columns.tolist())
 print("Feature matrix shape:", X.shape)
 print("Target distribution:")
 print(y.value_counts())
 Features: ['Age', 'Smokes', 'AreaQ', 'Alkhol']
 Feature matrix shape: (59, 4)
 Target distribution:
 Result
 0    31
 1    28
 Name: count, dtype: int64




 
 # Train-test split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 print("Training set size:", X_train.shape[0])
 print("Test set size:", X_test.shape[0])
 Training set size: 47
 Test set size: 12


 
 # Feature scaling
 scaler = StandardScaler()
 X_train_scaled = scaler.fit_transform(X_train)
 X_test_scaled = scaler.transform(X_test)
 print("Feature scaling completed")
 print("Scaled training data shape:", X_train_scaled.shape)
Feature scaling completed
 Scaled training data shape: (47, 4)


 
 # Building Decision Tree classifier
 dt_baseline = DecisionTreeClassifier(random_state=42)
 dt_baseline.fit(X_train_scaled, y_train)


 
 # Making predictions
 y_pred_baseline = dt_baseline.predict(X_test_scaled)
 print("Baseline Decision Tree model trained")
 Baseline Decision Tree model trained

 
 # Evaluating baseline model
 baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
 baseline_precision = precision_score(y_test, y_pred_baseline)
 baseline_recall = recall_score(y_test, y_pred_baseline)
 baseline_f1 = f1_score(y_test, y_pred_baseline)
 baseline_cm = confusion_matrix(y_test, y_pred_baseline)
 print("Baseline Model Performance:")
 print(f"Accuracy: {baseline_accuracy:.4f}")
 print(f"Precision: {baseline_precision:.4f}")
 print(f"Recall: {baseline_recall:.4f}")
 print(f"F1-Score: {baseline_f1:.4f}")
 print("\nConfusion Matrix:")
 print(baseline_cm)
 Baseline Model Performance:
 Accuracy: 1.0000
 Precision: 1.0000
 Recall: 1.0000
 F1-Score: 1.0000
 Confusion Matrix:
 [[6 0]
 [0 6]]

 
 # Feature importance analysis
 feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_baseline.feature_importances_
 }).sort_values('Importance', ascending=False)
 print("Feature Importance (Baseline Model):")
 print(feature_importance)
 Feature Importance (Baseline Model):
  Feature  Importance
 3  Alkhol    0.792048
 2   AreaQ    0.143861
 0     Age    0.064091
 1  Smokes    0.000000



 
 # Applying PCA to retain components explaining ≥95% variance
 pca_full = PCA()
 pca_full.fit(X_train_scaled)

 
 # Calculating cumulative explained variance
 cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

 
 # Finding number of components for ≥95% variance
 n_components = np.argmax(cumulative_variance >= 0.95) + 1
 print("PCA Analysis:")
 print(f"Explained variance ratios: {pca_full.explained_variance_ratio_}")
 print(f"Cumulative variance: {cumulative_variance}")
 print(f"Components needed for ≥95% variance: {n_components}")
 print(f"Actual variance retained: {cumulative_variance[n_components-1]:.4f}")
 PCA Analysis:
 Explained variance ratios: [0.53091571 0.25421908 0.15870353 0.05616168]
 Cumulative variance: [0.53091571 0.78513479 0.94383832 1.        ]
 Components needed for ≥95% variance: 4
 Actual variance retained: 1.0000


 
 # Applying PCA transformation
 pca = PCA(n_components=n_components, random_state=42)
 X_train_pca = pca.fit_transform(X_train_scaled)
 X_test_pca = pca.transform(X_test_scaled)
print(f"Original dimensions: {X_train_scaled.shape[1]}")
 print(f"PCA dimensions: {X_train_pca.shape[1]}")
 print(f"Variance retained: {pca.explained_variance_ratio_.sum():.4f}")
 Original dimensions: 4
 PCA dimensions: 4
 Variance retained: 1.0000



 
 # Building Decision Tree with PCA features
 dt_pca = DecisionTreeClassifier(random_state=42)
 dt_pca.fit(X_train_pca, y_train)



 
 # Making predictions
 y_pred_pca = dt_pca.predict(X_test_pca)
 print("PCA Decision Tree model trained")
 PCA Decision Tree model trained


 
 # Evaluating PCA model
 pca_accuracy = accuracy_score(y_test, y_pred_pca)
 pca_precision = precision_score(y_test, y_pred_pca)
 pca_recall = recall_score(y_test, y_pred_pca)
 pca_f1 = f1_score(y_test, y_pred_pca)
 pca_cm = confusion_matrix(y_test, y_pred_pca)
 print("PCA Model Performance:")
 print(f"Accuracy: {pca_accuracy:.4f}")
 print(f"Precision: {pca_precision:.4f}")
 print(f"Recall: {pca_recall:.4f}")
 print(f"F1-Score: {pca_f1:.4f}")
 print("\nConfusion Matrix:")
 print(pca_cm)
 PCA Model Performance:
 Accuracy: 0.9167
 Precision: 0.8571
 Recall: 1.0000
 F1-Score: 0.9231
 Confusion Matrix:
 [[5 1]
 [0 6]]


 
 # Comparing baseline and PCA models
 comparison = pd.DataFrame({
 'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
 'Baseline': [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1],
 'PCA': [pca_accuracy, pca_precision, pca_recall, pca_f1]
 })
 comparison['Difference'] = comparison['PCA'] - comparison['Baseline']
 print("Model Comparison:")
 print(comparison)
 Model Comparison:
      Metric  Baseline       PCA  Difference
 0   Accuracy       1.0  0.916667   -0.083333
 1  Precision       1.0  0.857143   -0.142857
 2     Recall       1.0  1.000000    0.000000
 3   F1-Score       1.0  0.923077   -0.076923



 
 # Discussion of results
 for idx, row in feature_importance.iterrows():
 if row['Importance'] > 0:
 print(f"   
{row['Feature']}: {row['Importance']:.4f}")
 print(f"   - Dimensions reduced from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")
 print(f"   - Variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
 accuracy_change = pca_accuracy - baseline_accuracy
 if accuracy_change > 0:
 print(f"   - Accuracy improved by {accuracy_change:.4f}")
 elif accuracy_change < 0:
 print(f"   - Accuracy decreased by {abs(accuracy_change):.4f}")
 else:
 print(f"   - Accuracy remained the same")
p
 (
 y
 )
 print(f"   - Baseline model accuracy: {baseline_accuracy:.4f}")
 print(f"   - PCA model accuracy: {pca_accuracy:.4f}")
 if pca_accuracy > baseline_accuracy:
 print(f"   - PCA improved performance while reducing dimensionality")
 elif pca_accuracy < baseline_accuracy:
 print(f"   - PCA reduced performance but simplified the model")
 else:
 print(f"   - PCA maintained performance with reduced complexity")
   Alkhol: 0.7920
   AreaQ: 0.1439
   Age: 0.0641
   - Dimensions reduced from 4 to 4
   - Variance retained: 100.00%
   - Accuracy decreased by 0.0833
   - Baseline model accuracy: 1.0000
   - PCA model accuracy: 0.9167
   - PCA reduced performance but simplified the model
