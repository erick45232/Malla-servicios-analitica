from flask import Flask, request, jsonify
from data_analyzer import DataAnalyzer

app = Flask(__name__)

# Ruta al archivo CSV
CSV_PATH = "2000.csv"

@app.route('/kmeans', methods=['POST'])
def ejecutar_kmeans():
    columnas = request.json.get('columnas')
    n_clusters = request.json.get('n_clusters', 3)

    analyzer = DataAnalyzer(CSV_PATH)
    analyzer.aplicar_kmeans(columnas, n_clusters)

    return jsonify({"mensaje": "KMeans ejecutado con éxito", "clusters": n_clusters})


@app.route('/knn', methods=['POST'])
def ejecutar_knn():
    columnas = request.json.get('columnas')
    target = request.json.get('target')
    k = request.json.get('k', 5)

    analyzer = DataAnalyzer(CSV_PATH)
    analyzer.aplicar_knn(columnas, target, k)

    return jsonify({"mensaje": "KNN ejecutado con éxito", "vecinos": k})


@app.route('/analisis_completo', methods=['POST'])
def ejecutar_ambos():
    columnas = request.json.get('columnas')
    target = request.json.get('target')
    n_clusters = request.json.get('n_clusters', 3)
    k = request.json.get('k', 5)

    analyzer = DataAnalyzer(CSV_PATH)
    analyzer.aplicar_kmeans(columnas, n_clusters)
    analyzer.aplicar_knn(columnas, target, k)

    return jsonify({
        "mensaje": "KMeans y KNN ejecutados con éxito",
        "clusters": n_clusters,
        "vecinos": k
    })


if __name__ == '__main__':
    app.run(debug=True)
