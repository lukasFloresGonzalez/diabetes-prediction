import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,average_precision_score


def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct='%1.1f%%')
    ax.axis('equal')

file_path = 'diabetes_prediction_dataset.csv'
data = pd.read_csv(file_path)

# Visualización básica de los datos
data.hist(bins=50, figsize=(20,15))
plt.show()

# Dividir el conjunto de datos por género
female_data = data[data['gender'] == 'Female']
male_data = data[data['gender'] == 'Male']

# Visualización básica de los datos con histogramas separados por género
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 15), sharex=True, sharey=True)

# Histograma para mujeres
axes[0].hist(female_data['age'], bins=50, alpha=0.7, color='pink', label='Female')
axes[0].set_title('Histograma de Edad para Mujeres')

# Histograma para hombres
axes[1].hist(male_data['age'], bins=50, alpha=0.7, color='blue', label='Male')
axes[1].set_title('Histograma de Edad para Hombres')

plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Identificando las columnas categóricas y numéricas
categorical_columns = ['gender', 'smoking_history']
numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

corr = data[numerical_columns].corr()
print(corr)

# Visualizar la matriz de correlación
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Matriz de Correlación')
plt.show()

# Creando transformadores para columnas numéricas y categóricas
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

############
# Aplicando el preprocesador al dataset
X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_processed = preprocessor.fit_transform(X)

# Obtener los nombres de las columnas después del preprocesamiento
processed_numerical_columns = (numerical_columns +
                               list(preprocessor.named_transformers_['cat']
                                    .get_feature_names_out(categorical_columns)))

################

modelo_rf = RandomForestClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
modelo_rf.fit(X_train, y_train)

print('Información del dataset:'
      '\n las clases del target son: {}'.format(Counter(y_train)))
plot_pie(y_train)

X_train_us, y_train_us = RandomUnderSampler(
    sampling_strategy='not minority', random_state=0
).fit_resample(X_train, y_train)

index_us = np.arange(len(X_train_us))
print('Información del dataset con Random Under Sampling:'
      '\n y: {}'.format(Counter(y_train_us)))
plot_pie(y_train_us)

sm = SMOTE(random_state=0,n_jobs=-1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print('Información del dataset aplicando el algoritmo Smote:'
      '\n y: {}'.format(Counter(y_train_res)))
plot_pie(y_train_res)

y_pred = modelo_rf.predict(X_test)

# Plotear la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Verdaderos')
plt.show()

# Aplicar PCA para obtener los componentes principales
pca = PCA()
X_pca = pca.fit_transform(X_processed)

# Plotear el screeplot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Screeplot')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulativa Explicada')
plt.grid(True)
plt.show()

# Obtener los dos primeros componentes principales para el biplot
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_processed)

# Entrenar el modelo (asegúrate de que esto se haga con tu conjunto de entrenamiento)
modelo_rf.fit(X_train, y_train)

# Obtener las importancias de las características y los índices ordenados
importances = modelo_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Adaptar los nombres de las características para que coincidan con las importancias ordenadas
# Si ya tienes un preprocesamiento hecho con ColumnTransformer y quieres los nombres originales:
feature_names = preprocessor.get_feature_names_out()

# Visualizar la importancia de las características con espaciado ajustado
plt.figure(figsize=(15, 5))
plt.title("Feature importances")
plt.bar(range(len(importances)), importances[indices], color="r", align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)

# Ajustar el espaciado
plt.subplots_adjust(bottom=0.3)

plt.xlim([-1, len(importances)])
plt.show()

# Biplot
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, palette='viridis')
plt.quiver(0, 0, pca.components_[0, 0], pca.components_[1, 0], color='r', scale=3, scale_units='xy')
plt.quiver(0, 0, pca.components_[0, 1], pca.components_[1, 1], color='r', scale=3, scale_units='xy')
plt.title('Biplot')
plt.xlabel('Primer Componente Principal')
plt.ylabel('Segundo Componente Principal')
plt.legend(title='Diabetes', loc='upper right', labels=['No Diabetes', 'Diabetes'])
plt.show()

print(classification_report(y_test, y_pred))