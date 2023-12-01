import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import numpy as np


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

# Creando transformadores para columnas numéricas y categóricas
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Aplicando el preprocesador al dataset
X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_processed = preprocessor.fit_transform(X)

modelo_rf = RandomForestClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
modelo_rf.fit(X_train, y_train)

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