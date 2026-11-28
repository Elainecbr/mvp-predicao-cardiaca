import math

FEATURES = (
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
)
CATEGORIES = {
    "Sex": {"M": 1, "F": 0},
    "ChestPainType": {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3},
    "RestingECG": {"Normal": 0, "ST": 1, "LVH": 2},
    "ExerciseAngina": {"N": 0, "Y": 1},
    "ST_Slope": {"Up": 0, "Flat": 1, "Down": 2},
}
NUMERIC_RANGES = {
    "Age": (1, 120),
    "RestingBP": (1, 300),
    "Cholesterol": (1, 1000),
    "FastingBS": (0, 1),
    "MaxHR": (1, 300),
    "Oldpeak": (-10, 10),
}


def encode_features(data):
    if not isinstance(data, dict) or set(data) != set(FEATURES):
        raise ValueError("Envie exatamente os 11 campos solicitados.")
    encoded = []
    for field in FEATURES:
        value = data[field]
        if field in CATEGORIES:
            if not isinstance(value, str) or value not in CATEGORIES[field]:
                raise ValueError(f"Valor inválido para {field}.")
            encoded.append(CATEGORIES[field][value])
        else:
            low, high = NUMERIC_RANGES[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
                or (field != "Oldpeak" and not float(value).is_integer())
            ):
                raise ValueError(f"Valor inválido para {field}.")
            encoded.append(value)
    return encoded
