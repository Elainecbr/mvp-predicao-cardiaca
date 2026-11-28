import csv
import json
import pickle
from pathlib import Path

import pytest
from sklearn.metrics import accuracy_score, recall_score

from app import create_app
from predicao import MODEL_PATH
from validacao import FEATURES, encode_features

DATA_DIR = Path(__file__).parent / "MachineLearning/data"
SAMPLE = {
    "Age": 40, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 140,
    "Cholesterol": 289, "FastingBS": 0, "RestingECG": "Normal",
    "MaxHR": 172, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
}


@pytest.fixture
def client():
    return create_app().test_client()


def test_encoding_matches_notebook():
    assert encode_features(SAMPLE) == [40, 1, 0, 140, 289, 0, 0, 172, 0, 0.0, 0]


@pytest.mark.parametrize(("field", "value", "index", "expected"), [
    ("Sex", "M", 1, 1), ("Sex", "F", 1, 0),
    ("ChestPainType", "ATA", 2, 0), ("ChestPainType", "NAP", 2, 1),
    ("ChestPainType", "ASY", 2, 2), ("ChestPainType", "TA", 2, 3),
    ("RestingECG", "Normal", 6, 0), ("RestingECG", "ST", 6, 1),
    ("RestingECG", "LVH", 6, 2),
    ("ExerciseAngina", "N", 8, 0), ("ExerciseAngina", "Y", 8, 1),
    ("ST_Slope", "Up", 10, 0), ("ST_Slope", "Flat", 10, 1),
    ("ST_Slope", "Down", 10, 2),
])
def test_each_category_matches_notebook(field, value, index, expected):
    features = encode_features({**SAMPLE, field: value})
    assert features[index] == expected


@pytest.mark.parametrize(("field", "value"), [
    ("Age", 1), ("Age", 120), ("RestingBP", 1), ("RestingBP", 300),
    ("Cholesterol", 1), ("Cholesterol", 1000), ("FastingBS", 0),
    ("FastingBS", 1), ("MaxHR", 1), ("MaxHR", 300),
    ("Oldpeak", -10), ("Oldpeak", 10), ("Oldpeak", 1.5),
])
def test_numeric_boundaries_are_accepted(field, value):
    encoded = encode_features({**SAMPLE, field: value})
    assert encoded[FEATURES.index(field)] == value


@pytest.mark.parametrize(("field", "value"), [
    ("Age", 0), ("Age", 121), ("RestingBP", 301), ("Cholesterol", 0),
    ("Cholesterol", 1001), ("FastingBS", -1), ("MaxHR", 0),
    ("MaxHR", 301), ("Oldpeak", -10.1), ("Oldpeak", 10.1),
    ("Oldpeak", float("nan")), ("Oldpeak", float("inf")),
    ("Sex", None), ("ChestPainType", "unexpected"), ("RestingECG", "unexpected"),
    ("ExerciseAngina", "unexpected"), ("ST_Slope", "unexpected"),
])
def test_invalid_boundaries_and_categories_are_rejected(field, value):
    with pytest.raises(ValueError):
        encode_features({**SAMPLE, field: value})


def test_feature_order_is_independent_of_json_order():
    reversed_input = dict(reversed(list(SAMPLE.items())))
    assert encode_features(reversed_input) == encode_features(SAMPLE)


@pytest.mark.parametrize(("data", "expected"), [
    (SAMPLE, 0),
    ({
        "Age": 65, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 145,
        "Cholesterol": 250, "FastingBS": 1, "RestingECG": "ST",
        "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.0, "ST_Slope": "Flat",
    }, 1),
])
def test_known_predictions_are_preserved(client, data, expected):
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json == {"prediction": expected}


def test_predict_matches_model(client):
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    expected = int(model.predict([encode_features(SAMPLE)])[0])
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.json == {"prediction": expected}


@pytest.mark.parametrize("changes", [
    {"Age": True}, {"Age": 40.5}, {"Sex": "X"}, {"Oldpeak": "1.5"},
    {"RestingBP": 0}, {"FastingBS": 2}, {"extra": "nome"},
])
def test_invalid_input_is_rejected(client, changes):
    response = client.post("/predict", json={**SAMPLE, **changes})
    assert response.status_code == 400
    assert "error" in response.json


def test_missing_field_and_wrong_content_type(client):
    assert client.post("/predict", json={"Age": 40}).status_code == 400
    assert client.post("/predict", data="test").status_code == 415


@pytest.mark.parametrize("data", [[], None, "text", 42])
def test_json_must_be_an_object_with_all_fields(client, data):
    response = client.post("/predict", data=json.dumps(data),
                           content_type="application/json")
    assert response.status_code == 400


@pytest.mark.parametrize("origin", [
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:8080", "http://127.0.0.1:8080",
])
def test_allowed_frontend_origins(client, origin):
    response = client.post("/predict", json=SAMPLE, headers={"Origin": origin})
    assert response.headers["Access-Control-Allow-Origin"] == origin


def test_unlisted_origin_is_not_allowed(client):
    response = client.post("/predict", json=SAMPLE, headers={"Origin": "http://example.com"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_model_against_saved_test_split():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with (DATA_DIR / "X_test_cardiaco.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    with (DATA_DIR / "y_test_cardiaco.csv").open(newline="") as file:
        targets = [int(row["HeartDisease"]) for row in csv.DictReader(file)]
    assert list(rows[0]) == list(FEATURES)
    predictions = model.predict([[float(row[name]) for name in FEATURES] for row in rows])
    assert accuracy_score(targets, predictions) >= 0.80
    assert recall_score(targets, predictions) >= 0.80
