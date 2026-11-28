import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

from predicao import carregar_pipeline, prever
from validacao import encode_features


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    CORS(app, resources={r"/predict": {"origins": [
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8080", "http://127.0.0.1:8080",
    ]}})

    model = carregar_pipeline()

    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Envie os dados no formato JSON."), 415
        data = request.get_json()
        try:
            features = encode_features(data)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        try:
            prediction = prever(model, features)
        except Exception:
            app.logger.exception("Falha ao executar a predição do modelo.")
            return jsonify(error="Não foi possível executar a predição."), 500
        if prediction not in (0, 1):
            app.logger.error("O modelo retornou uma classe inesperada: %s", prediction)
            return jsonify(error="O modelo retornou um resultado inesperado."), 500
        return jsonify(prediction=prediction)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=5000)
