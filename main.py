import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from tabulate import tabulate
import seaborn as sns

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.describe())
    # pairplot
    sns.pairplot(df, hue='class', diag_kind='kde')
    plt.show()

    # heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    X = df.drop("class", axis=1)
    y = df["class"]
    return X, y

def standardize_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_and_evaluate_classifier(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, cm

def plot_feature_importance(features, importances, title="Importancia de las características"):
    sorted_importances = np.argsort(importances)
    plt.barh(features[sorted_importances][-10:], importances[sorted_importances][-10:])
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm, title="Matriz de Confusión"):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(title)
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.show()

def main():
    file_path = "indians_diabetes.csv"
    X, y= load_data(file_path)

    # 1. Estandarizar características
    X_scaled = standardize_features(X)

    # Plotear importancia de las características al principio
    clf_initial = RandomForestClassifier(n_estimators=1000)
    clf_initial.fit(X_scaled, y)
    plot_feature_importance(X.columns, clf_initial.feature_importances_, title="Importancia de las características (Inicial)")

    # 2. Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y)

    # 3. Entrenar y evaluar clasificadores con n=100 y n=1000
    for n_estimators in [100, 1000]:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        accuracy, precision, recall, f1, cm = train_and_evaluate_classifier(clf, X_train, X_test, y_train, y_test)

        print(f"Resultados con n={n_estimators}:")
        print("Confusion Matrix:")
        print(cm)

        result_table = [
            ["Accuracy", accuracy],
            ["Precision", precision],
            ["Recall", recall],
            ["F1 Score", f1]
        ]
        print(tabulate(result_table, headers=["Metrica", "Valor"], tablefmt="pretty"))

        # Plot Confusion Matrix
        plot_confusion_matrix(cm, title=f"Matriz de Confusión (n={n_estimators})")

        # Validación cruzada para una evaluación más robusta
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        print(f"Validación cruzada (n={n_estimators}) - Accuracy: {np.mean(cv_scores)}")

    # 4. Entrenar modelo con características más informativas usando SelectFromModel
    sfm = SelectFromModel(clf)
    sfm.fit(X_scaled, y)
    X_selected = sfm.transform(X_scaled)

    X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(
        X_selected, y, test_size=0.1, stratify=y
    )

    clf_selected = RandomForestClassifier(n_estimators=1000)
    accuracy_selected, precision_selected, recall_selected, f1_selected, cm_selected = train_and_evaluate_classifier(
        clf_selected, X_train_selected, X_test_selected, y_train_selected, y_test_selected)

    print("Resultados con características seleccionadas:")
    result_table_2 = [
        ["Accuracy", accuracy_selected],
        ["Precision", precision_selected],
        ["Recall", recall_selected],
        ["F1 Score", f1_selected]
    ]
    print(tabulate(result_table_2, headers=["Metrica", "Valor"], tablefmt="pretty"))

    print(cm_selected)
    # Plot Confusion Matrix for Selected Features
    plot_confusion_matrix(cm_selected, title="Matriz de Confusión (con características seleccionadas)")

if __name__ == "__main__":
    main()
