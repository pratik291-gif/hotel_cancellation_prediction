import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

def evaluate_model(model_path="model.pkl", test_data_path="test_data.pkl", output_dir="eval_outputs"):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load model and data
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {auc:.4f}\n")
        
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Feature Importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        indices = importances.argsort()[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_test.shape[1]), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png")
        plt.close()

if __name__ == "__main__":
    evaluate_model()
