import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, make_scorer, f1_score, mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN, SMOTETomek

def load_and_preprocess_data():
    data = pd.read_csv("data/dataset.csv")
    X = data.drop(columns=["id", "credit_score"])
    y = data["credit_score"].values
    return X, y

def create_feature_engineering(X, y):
    X = X.copy()
    # Korelasyon ile en yüksek iki özelliğin oranı, çarpımı, farkı
    corr_with_target = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
    top2 = corr_with_target.index[:2]
    X['feature_ratio'] = X[top2[0]] / (X[top2[1]] + 1e-5)
    X['feature_product'] = X[top2[0]] * X[top2[1]]
    X['feature_diff'] = X[top2[0]] - X[top2[1]]
    # Logaritmik özellikler (pozitif olanlar için)
    for col in X.columns:
        if (X[col] > 0).all():
            X[f'log_{col}'] = np.log1p(X[col])
    # Toplam, ortalama, std
    X['feature_sum'] = X.sum(axis=1)
    X['feature_mean'] = X.mean(axis=1)
    X['feature_std'] = X.std(axis=1)
    # En iyi 15 özelliği seç (RandomForest ile)
    rf_fs = RandomForestClassifier(random_state=42)
    rf_fs.fit(X, y)
    importances = rf_fs.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15
    top_features = X.columns[indices[:top_n]]
    print("En iyi 15 özellik:", list(top_features))
    X = X[top_features].copy()
    return X

def custom_scorer(y_true, y_pred_proba):
    # Optimal threshold'u bul
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    
    # Optimal threshold ile tahminler
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    return accuracy_score(y_true, y_pred)

def find_best_params(X_train, y_train):
    # GridSearchCV için parametre aralıkları
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 3, 5]  # Sınıf dengesizliği için
    }
    
    # Base model
    model = XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Custom scorer kullan
    scorer = make_scorer(custom_scorer, needs_proba=True)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("Grid Search başlatılıyor...")
    grid_search.fit(X_train, y_train)
    print("\nEn iyi parametreler:", grid_search.best_params_)
    print("En iyi skor:", grid_search.best_score_)
    
    return grid_search.best_params_

def train_optimized_xgboost(X_train, X_test, y_train, y_test, params=None):
    if params is None:
        params = {
            'learning_rate': 0.2,
            'max_depth': 3,
            'n_estimators': 200,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42
        }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return model, y_pred, y_pred_proba

def find_optimal_threshold_f1(y_true, y_pred_proba):
    thresholds = np.linspace(0, 1, 200)
    f1_scores = [f1_score(y_true, (y_pred_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def evaluate_model_with_threshold(y_test, y_pred_proba, threshold=0.5):
    # Verilen threshold ile tahminleri yap
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"\n--- Model Sonuçları (Threshold: {threshold:.3f}) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek Değerler')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, X):
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(15))
    plt.title('En Önemli 15 Özellik')
    plt.savefig('feature_importance.png')
    plt.close()
    return importance

def plot_roc_curve(y_test, y_pred_proba, optimal_threshold):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Optimal noktayı işaretle
    optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', 
             label=f'Optimal threshold ({optimal_threshold:.3f})')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

def remove_outliers_iqr(X, y):
    # IQR yöntemiyle aykırı değerleri kaldır
    X_clean = X.copy()
    mask = np.ones(len(X_clean), dtype=bool)
    for col in X_clean.select_dtypes(include=[np.number]).columns:
        Q1 = X_clean[col].quantile(0.25)
        Q3 = X_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask &= (X_clean[col] >= lower) & (X_clean[col] <= upper)
    return X_clean[mask], y[mask]

def main():
    X, y = load_and_preprocess_data()
    # Feature engineering (örnek: oran, çarpım, fark, log, toplam, ortalama, std)
    X = X.copy()
    X['feature_sum'] = X.sum(axis=1)
    X['feature_mean'] = X.mean(axis=1)
    X['feature_std'] = X.std(axis=1)
    for col in X.columns:
        if (X[col] > 0).all():
            X[f'log_{col}'] = np.log1p(X[col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n--- XGBoostRegressor ile Kredi Skoru Tahmini ---")
    model = XGBRegressor(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.4f}")

    # Özellik önem grafiği
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nEn önemli 10 özellik:")
    print(importance.head(10))
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(10))
    plt.title('En Önemli 10 Özellik')
    plt.savefig('feature_importance_regression.png')
    plt.close()
    print("\nGörselleştirme 'feature_importance_regression.png' olarak kaydedildi.")

if __name__ == "__main__":
    main() 