# ==========================================================
# main.py - API de Classificacao de Preco Airbnb
# ==========================================================
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lime import lime_tabular
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ----------------------------------------------------------
# Inicializacao do app
# ----------------------------------------------------------
app = FastAPI(
    title="API - Classificacao de Preco Airbnb",
    description="Classifica anuncios Airbnb em: baixo, medio ou luxo.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Carregar modelos e objetos
# ----------------------------------------------------------
model = joblib.load("models/modelo_airbnb.pkl")
scaler = joblib.load("models/scaler_airbnb.pkl")
le = joblib.load("models/label_encoder.pkl")
le_neigh = joblib.load("models/label_encoder_neighbourhood.pkl")
le_prop = joblib.load("models/label_encoder_property.pkl")
EXPECTED_FEATURES = tuple(scaler.feature_names_in_)

# ----------------------------------------------------------
# Constantes e funcoes de apoio
# ----------------------------------------------------------
# Mapeia respostas textuais do host para as colunas one-hot usadas pelo modelo.
HOST_RESPONSE_TIME_COLUMNS = {
    "a few days or more": "host_response_time_a few days or more",
    "unknown": "host_response_time_unknown",
    "within a day": "host_response_time_within a day",
    "within a few hours": "host_response_time_within a few hours",
    "within an hour": "host_response_time_within an hour",
}

# Mapeia tipos de quarto para as colunas derivadas durante o treinamento.
ROOM_TYPE_COLUMNS = {
    "entire home apt": "room_type_Entire home/apt",
    "hotel room": "room_type_Hotel room",
    "private room": "room_type_Private room",
    "shared room": "room_type_Shared room",
}

# Campos booleanos que precisam ser convertidos para 0/1 antes de alimentar o modelo.
BOOL_FIELDS = ("host_has_profile_pic", "host_identity_verified", "has_availability")

_NORMALIZE_TRANS = str.maketrans({"/": " ", "_": " ", "-": " "})
LIME_BACKGROUND_PATH = Path("models/lime_background.npy")
LIME_NUM_FEATURES = 20


def _normalize_text(value: str) -> str:
    """Normaliza texto para facilitar comparacoes sem depender de acentos ou pontuacao."""
    return " ".join(value.lower().translate(_NORMALIZE_TRANS).split())


# Constroi lookup com textos normalizados para aceitar variantes de entrada.
HOST_RESPONSE_TIME_LOOKUP = {}
for key, column in HOST_RESPONSE_TIME_COLUMNS.items():
    normalized = _normalize_text(key)
    HOST_RESPONSE_TIME_LOOKUP[normalized] = column
    HOST_RESPONSE_TIME_LOOKUP[_normalize_text(f"host response time {key}")] = column


# Idem para os tipos de quarto.
ROOM_TYPE_LOOKUP = {}
for key, column in ROOM_TYPE_COLUMNS.items():
    normalized = _normalize_text(key)
    ROOM_TYPE_LOOKUP[normalized] = column
    ROOM_TYPE_LOOKUP[_normalize_text(f"room type {key}")] = column


def _load_lime_background() -> np.ndarray:
    """Carrega amostras de background para o LIME ou cria fallback a partir do scaler."""
    if LIME_BACKGROUND_PATH.exists():
        try:
            # Usa amostras salvas para manter consistencia entre chamadas.
            background = np.load(LIME_BACKGROUND_PATH)
            if background.ndim == 2 and background.shape[1] == len(EXPECTED_FEATURES):
                return background
        except Exception:
            pass

    # Tenta gerar amostras sinteticas usando estatisticas do scaler (quando disponiveis)
    try:
        # Caso nao haja arquivo, tenta sintetizar a partir das estatisticas do scaler.
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        if mean is not None:
            raw = [np.asarray(mean)]
            if scale is not None:
                scale_array = np.asarray(scale)
                raw.append(mean + scale_array)
                raw.append(mean - scale_array)
            df = pd.DataFrame(raw, columns=EXPECTED_FEATURES)
            generated = scaler.transform(df)
            if generated.ndim == 2 and generated.shape[1] == len(EXPECTED_FEATURES):
                return generated
    except Exception:
        pass

    # Fallback: matriz de zeros na escala transformada (equivalente ao valor medio)
    fallback_rows = max(50, len(EXPECTED_FEATURES))
    return np.zeros((fallback_rows, len(EXPECTED_FEATURES)))


# Precarrega amostras para reutilizar nas explicacoes LIME.
LIME_BACKGROUND = _load_lime_background()

# Mantem uma unica instancia do LIME para reduzir latencia e garantir consistencia.
LIME_EXPLAINER = lime_tabular.LimeTabularExplainer(
    training_data=LIME_BACKGROUND,
    feature_names=list(EXPECTED_FEATURES),
    class_names=le.classes_,
    mode="classification",
    random_state=42,
)

def _resolve_host_response_time(value: str) -> str:
    """Mapeia o texto do tempo de resposta para a coluna usada no treinamento."""
    normalized = _normalize_text(value)
    column = HOST_RESPONSE_TIME_LOOKUP.get(normalized)
    if not column and normalized.startswith("host response time "):
        # Permite entradas com o prefixo completo enviado por alguns clientes.
        column = HOST_RESPONSE_TIME_LOOKUP.get(normalized.replace("host response time ", "", 1))
    if not column:
        raise HTTPException(
            status_code=422,
            detail=f"Valor de host_response_time nao suportado: '{value}'.",
        )
    return column


def _resolve_room_type(value: str) -> str:
    """Mapeia o texto do tipo de quarto para a coluna usada no treinamento."""
    normalized = _normalize_text(value)
    column = ROOM_TYPE_LOOKUP.get(normalized)
    if not column and normalized.startswith("room type "):
        column = ROOM_TYPE_LOOKUP.get(normalized.replace("room type ", "", 1))
    if not column:
        raise HTTPException(
            status_code=422,
            detail=f"Valor de room_type nao suportado: '{value}'.",
        )
    return column


def _encode_category(encoder, value: str, field_name: str) -> int:
    """Codifica categorias tratando diferencas simples de capitalizacao."""
    cleaned = value.strip()
    if cleaned in encoder.classes_:
        return int(encoder.transform([cleaned])[0])

    lower_cleaned = cleaned.lower()
    for original in encoder.classes_:
        # Normaliza caixa para evitar rejeitar textos com capitalizacao diferente.
        if original.lower() == lower_cleaned:
            return int(encoder.transform([original])[0])

    raise HTTPException(
        status_code=422,
        detail=f"Valor '{value}' nao foi visto durante o treinamento para '{field_name}'.",
    )


def _prepare_features(payload: "AirbnbInput") -> pd.DataFrame:
    """Transforma a carga util crua no vetor de atributos esperado pelo modelo."""
    # Garante que trabalharemos com um dicionario editavel.
    raw = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()

    # Separa categorias que precisam de tratamento especial antes do dataframe.
    host_response_time = raw.pop("host_response_time")
    room_type = raw.pop("room_type")
    neighbourhood = raw.pop("neighbourhood_cleansed")
    property_type = raw.pop("property_type")

    # Converte booleanos para 0/1 porque o modelo foi treinado com floats.
    for field in BOOL_FIELDS:
        raw[field] = float(raw[field])

    # Reproduz feature derivada usada durante o treinamento.
    raw["bath_per_bedroom"] = raw["bathrooms"] / (raw["bedrooms"] + 1)

    # Aplica label encoding igual ao pipeline de treinamento.
    raw["neighbourhood_cleansed_encoded"] = _encode_category(
        le_neigh, neighbourhood, "neighbourhood_cleansed"
    )
    raw["property_type_encoded"] = _encode_category(
        le_prop, property_type, "property_type"
    )

    # Inicializa as variaveis dummies de tempo de resposta e ativa a recebida.
    raw.update({col: 0.0 for col in HOST_RESPONSE_TIME_COLUMNS.values()})
    raw[_resolve_host_response_time(host_response_time)] = 1.0

    # Idem para o tipo de quarto.
    raw.update({col: 0.0 for col in ROOM_TYPE_COLUMNS.values()})
    raw[_resolve_room_type(room_type)] = 1.0

    df = pd.DataFrame([raw])
    # Reordena e garante que colunas ausentes sejam preenchidas com zero.
    return df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

# Etiquetas legiveis e agrupamentos para apresentar o resultado do LIME.
NICE_FEATURES = {
    "latitude": ("Latitude", "Localizacao"),
    "longitude": ("Longitude", "Localizacao"),
    "accommodates": ("Capacidade (hospedes)", "Tamanho"),
    "bathrooms": ("Banheiros", "Tamanho"),
    "bedrooms": ("Quartos", "Tamanho"),
    "beds": ("Camas", "Tamanho"),
    "amenities_count": ("Qtd. comodidades", "Tamanho"),
    "host_acceptance_rate": ("Taxa de aceitacao do host", "Host"),
    "host_response_rate": ("Taxa de resposta do host", "Host"),
    "host_days_active": ("Dias ativo (host)", "Host"),
    "bath_per_bedroom": ("Banheiros por quarto", "Tamanho"),
    "room_type_Entire home/apt": ("Tipo do quarto = Apto inteiro", "Tipo"),
    "room_type_Private room": ("Tipo do quarto = Quarto priv.", "Tipo"),
    "room_type_Shared room": ("Tipo do quarto = Quarto comp.", "Tipo"),
    "room_type_Hotel room": ("Tipo do quarto = Quarto de hotel", "Tipo"),
    "host_response_time_within an hour": ("Tempo de resp. = \u22641h", "Host"),
    "host_response_time_within a few hours": ("Tempo de resp. = poucas horas", "Host"),
    "host_response_time_within a day": ("Tempo de resp. = \u22641 dia", "Host"),
    "host_response_time_a few days or more": ("Tempo de resp. = varios dias", "Host"),
    "host_response_time_unknown": ("Tempo de resp. = desconhecido", "Host"),
}
 

def _inverse_scale_row(scaler_obj, x_scaled_row, feature_names):
    """Reverte uma linha escalonada para valores originais aproximados."""
    mean = getattr(scaler_obj, "mean_", None)
    var = getattr(scaler_obj, "var_", None)
    if mean is None or var is None:
        return dict(zip(feature_names, x_scaled_row))
    scale = np.sqrt(var)
    original = x_scaled_row * scale + mean
    return dict(zip(feature_names, original))


def build_lime_readable(exp, x_scaled_row, df_row_original, predicted_label, topk=8, min_abs=0.02):
    """Organiza a saida do LIME em um formato amigavel para retorno JSON."""
    available = exp.available_labels()
    if isinstance(predicted_label, str):
        try:
            label_idx = list(exp.class_names).index(predicted_label)
        except ValueError:
            label_idx = available[0]
    else:
        label_idx = available[0]

    # Seleciona o conjunto de pesos referente ao rotulo previsto.
    pairs = exp.as_map().get(label_idx, [])
    if not pairs and available:
        pairs = exp.as_map().get(available[0], [])

    # Recupera valores na escala original para mostrar ao usuario.
    original_vals = _inverse_scale_row(scaler, x_scaled_row, df_row_original.columns)

    # Ordena as contribuicoes por importancia absoluta.
    pairs = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)
    total = sum(abs(weight) for _, weight in pairs) or 1.0
    covered = 0.0
    itens = []

    for idx, weight in pairs:
        if abs(weight) < min_abs:
            continue

        name = exp.domain_mapper.feature_names[idx]
        label, group = NICE_FEATURES.get(name, (name, "Outros"))

        raw_val = df_row_original.iloc[0].get(name, original_vals.get(name))
        if name.startswith("room_type_") or name.startswith("host_response_time_"):
            valor = "sim" if (raw_val is not None and raw_val >= 0.5) else "nao"
        elif isinstance(raw_val, (int, float, np.floating, np.integer)):
            valor = round(float(raw_val), 2)
        else:
            valor = raw_val

        # Direcao indica se a feature empurra a probabilidade para a classe prevista.
        direcao = f"\u2191 p/ {predicted_label}" if weight > 0 else f"\u2193 p/ {predicted_label}"

        itens.append(
            {
                "feature": name,
                "rotulo": label,
                "grupo": group,
                "valor": valor,
                "impacto": round(float(weight), 3),
                "direcao": direcao,
            }
        )

        covered += abs(weight)
        if len(itens) >= topk:
            break

    cobertura = round(100.0 * covered / total, 1)

    return {
        "itens": itens,
        "cobertura_pct": cobertura,
    }


# ----------------------------------------------------------
# Modelo de entrada (request body)
# ----------------------------------------------------------
class AirbnbInput(BaseModel):
    host_response_rate: float
    host_acceptance_rate: float
    host_listings_count: float
    host_total_listings_count: float
    host_has_profile_pic: bool
    host_identity_verified: bool
    latitude: float
    longitude: float
    accommodates: int
    bathrooms: float
    bedrooms: float
    beds: float
    minimum_nights: int
    maximum_nights: int
    minimum_minimum_nights: float
    maximum_minimum_nights: float
    minimum_maximum_nights: float
    maximum_maximum_nights: float
    minimum_nights_avg_ntm: float
    maximum_nights_avg_ntm: float
    has_availability: bool
    host_days_active: int
    amenities_count: int
    host_response_time: str
    room_type: str
    neighbourhood_cleansed: str
    property_type: str

    class Config:
        anystr_strip_whitespace = True


# ----------------------------------------------------------
# Funcao de previsao
# ----------------------------------------------------------
@app.post("/predict")
def predict(data: AirbnbInput):
    """
    Faz a previsao da classe de preco (baixo, medio, luxo)
    e retorna tambem a explicacao LIME das principais variaveis.
    """
    try:
        df = _prepare_features(data)

        # Escalonar
        X_scaled = scaler.transform(df)

        # Previsao
        pred_num = model.predict(X_scaled)[0]
        classe_prevista = le.inverse_transform([pred_num])[0]
        prob = model.predict_proba(X_scaled)[0]

        # ---------- LIME ----------
        # Reutiliza o explicador global para reduzir custo por chamada.
        exp = LIME_EXPLAINER.explain_instance(
            data_row=X_scaled[0],
            predict_fn=model.predict_proba,
            num_features=LIME_NUM_FEATURES,
        )
        explicacao_legivel = build_lime_readable(
            exp,
            X_scaled[0],
            df,
            classe_prevista,
            topk=10,
            min_abs=0.02,
        )

        resultado = {
            "classe_prevista": classe_prevista,
            "confianca": f"{prob[pred_num]*100:.1f}%",
            "probabilidades": {
                c: f"{p*100:.1f}%" for c, p in zip(le.classes_, prob)
            },
            "explicacao_LIME": explicacao_legivel,
        }

        return {"status": "ok", "resultado": resultado}

    except HTTPException as exc:
        raise exc
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": str(e),
        }


# ----------------------------------------------------------
# Teste rapido da API
# ----------------------------------------------------------
@app.get("/")
def home():
    return {"status": "online", "mensagem": "API Airbnb pronta para previsoes."}
