import pickle
import warnings
from pathlib import Path

from sklearn.exceptions import InconsistentVersionWarning

from validacao import FEATURES

MODEL_PATH = Path(__file__).parent / "MachineLearning/pipelines/modelo_cardiaco_pipeline.pkl"


def carregar_pipeline():
    # Carregue somente o modelo do projeto, nunca arquivos enviados pelo usuário.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with MODEL_PATH.open("rb") as model_file:
            model = pickle.load(model_file)
    if getattr(model, "n_features_in_", None) != len(FEATURES):
        raise ValueError("O modelo não aceita as 11 variáveis esperadas.")
    return model


def prever(model, features):
    return int(model.predict([features])[0])
