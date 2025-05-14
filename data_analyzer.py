# --- CLASE ANALIZADOR ---
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

class DataAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.read_csv(filepath)
        self.results_dir = "resultados"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Archivo cargado con {self.data.shape[0]} filas y {self.data.shape[1]} columnas")

    def aplicar_kmeans(self, columnas, n_clusters=3, output_name="kmeans_result.csv"):
        X = self.data[columnas].dropna()
        modelo = KMeans(n_clusters=n_clusters, random_state=42)
        self.data.loc[X.index, "KMeans_Cluster"] = modelo.fit_predict(X)
        self.data.to_csv(os.path.join(self.results_dir, output_name), index=False)
        print(f"KMeans aplicado. Resultado guardado en: {output_name}")

    def aplicar_knn(self, columnas, target_col, k=5, output_name="knn_result.csv"):
        df = self.data.dropna(subset=columnas + [target_col])
        X = df[columnas]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        reporte = classification_report(y_test, y_pred, output_dict=True)
        reporte_df = pd.DataFrame(reporte).transpose()
        reporte_df.to_csv(os.path.join(self.results_dir, output_name))
        print(f"KNN aplicado. Reporte guardado en: {output_name}")

    def mostrar_columnas(self):
        print("Columnas disponibles:")
        print(self.data.columns.tolist())

    def ver_datos(self, n=5):
        print(self.data.head(n))
