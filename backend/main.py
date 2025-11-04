# ==========================================================
# main.py - API de Classificacao de Preco Airbnb
# ==========================================================
import json
import logging
import os
from pathlib import Path
import shap


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from lime import lime_tabular
from pydantic import BaseModel, Field, model_validator, ConfigDict
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
# Permite configurar o diretorio de artefatos por variavel de ambiente.
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = BACKEND_DIR.parent / "models"
MODELS_DIR = Path(os.getenv("AIRBNB_MODELS_DIR", DEFAULT_MODELS_DIR))

MODEL_PATH = MODELS_DIR / "modelo_airbnb.pkl"
SCALER_PATH = MODELS_DIR / "scaler_airbnb.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LABEL_ENCODER_NEIGH_PATH = MODELS_DIR / "label_encoder_neighbourhood.pkl"
LABEL_ENCODER_PROP_PATH = MODELS_DIR / "label_encoder_property.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
le_neigh = joblib.load(LABEL_ENCODER_NEIGH_PATH)
le_prop = joblib.load(LABEL_ENCODER_PROP_PATH)
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
LIME_BACKGROUND_PATH = MODELS_DIR / "lime_background.npy"
LIME_NUM_FEATURES = 20

logger = logging.getLogger("airbnb_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _log_event(event: str, **payload) -> None:
    """Registra eventos estruturados em formato JSON."""
    try:
        message = json.dumps({"event": event, **payload}, ensure_ascii=False, default=str)
    except TypeError:
        message = f"{event} | {payload}"
    logger.info(message)


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

# ----------------------------------------------------------
# SHAP - explicabilidade
# ----------------------------------------------------------
try:
    SHAP_EXPLAINER = shap.TreeExplainer(model)
    SHAP_BACKGROUND = None
except Exception:
    SHAP_BACKGROUND = LIME_BACKGROUND
    SHAP_EXPLAINER = shap.KernelExplainer(model.predict_proba, SHAP_BACKGROUND)

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
    original_payload = raw.copy()
    _log_event(
        "prepare_features_start",
        host_response_time=original_payload.get("host_response_time"),
        room_type=original_payload.get("room_type"),
        bool_fields={field: original_payload.get(field) for field in BOOL_FIELDS},
    )

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
    df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

    transformed_row = df.iloc[0].to_dict()
    non_zero_features = {
        name: value for name, value in transformed_row.items() if value not in (0, 0.0)
    }
    _log_event(
        "prepare_features_end",
        total_features=len(transformed_row),
        non_zero_features=non_zero_features,
    )
    return df

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


def _select_shap_row(shap_values, predicted_index: int):
    """Extrai o vetor SHAP referente a classe prevista, independente do formato retornado."""
    explanation_cls = getattr(shap, "Explanation", None)
    if explanation_cls is not None and isinstance(shap_values, explanation_cls):
        values = shap_values.values
        if values.ndim == 3:
            return values[0, predicted_index, :]
        if values.ndim == 2:
            return values[0]
        return values[predicted_index]

    if isinstance(shap_values, list):
        selected = shap_values[predicted_index]
        selected = np.asarray(selected)
        if selected.ndim == 2:
            return selected[0]
        return selected

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[0, predicted_index, :]
    return values[0]


def build_shap_readable(
    shap_values_row,
    x_scaled_row,
    df_row_original,
    predicted_label,
    topk: int = 10,
    min_abs: float = 0.0,
):
    """
    Organiza a saida do SHAP em um formato amigavel para retorno JSON.
    `shap_values_row` deve ser um array 1D com o valor SHAP de cada feature.
    """
    shap_array = np.asarray(shap_values_row).ravel()
    original_vals = _inverse_scale_row(scaler, x_scaled_row, df_row_original.columns)

    pairs = list(enumerate(shap_array))
    pairs = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)

    itens = []
    total = sum(abs(val) for _, val in pairs) or 1.0
    covered = 0.0

    for idx, impacto in pairs:
        if abs(impacto) < min_abs:
            continue

        name = df_row_original.columns[idx]
        label, group = NICE_FEATURES.get(name, (name, "Outros"))

        raw_val = df_row_original.iloc[0].get(name, original_vals.get(name))
        if name.startswith("room_type_") or name.startswith("host_response_time_"):
            valor = "sim" if (raw_val is not None and raw_val >= 0.5) else "nao"
        elif isinstance(raw_val, (int, float, np.floating, np.integer)):
            valor = round(float(raw_val), 2)
        else:
            valor = raw_val

        direcao = f"\u2191 p/ {predicted_label}" if impacto > 0 else f"\u2193 p/ {predicted_label}"

        itens.append(
            {
                "feature": name,
                "rotulo": label,
                "grupo": group,
                "valor": valor,
                "impacto": round(float(impacto), 3),
                "direcao": direcao,
            }
        )

        covered += abs(impacto)
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
    host_response_rate: float = Field(..., ge=0, le=100)
    host_acceptance_rate: float = Field(..., ge=0, le=100)
    host_listings_count: float = Field(..., ge=0)
    host_total_listings_count: float = Field(..., ge=0)
    host_has_profile_pic: bool
    host_identity_verified: bool
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accommodates: int = Field(..., ge=1)
    bathrooms: float = Field(..., ge=0)
    bedrooms: float = Field(..., ge=0)
    beds: float = Field(..., ge=0)
    minimum_nights: int = Field(..., ge=1)
    maximum_nights: int = Field(..., ge=1)
    minimum_minimum_nights: float = Field(..., ge=0)
    maximum_minimum_nights: float = Field(..., ge=0)
    minimum_maximum_nights: float = Field(..., ge=0)
    maximum_maximum_nights: float = Field(..., ge=0)
    minimum_nights_avg_ntm: float = Field(..., ge=0)
    maximum_nights_avg_ntm: float = Field(..., ge=0)
    has_availability: bool
    host_days_active: int = Field(..., ge=0)
    amenities_count: int = Field(..., ge=0)
    host_response_time: str
    room_type: str
    neighbourhood_cleansed: str
    property_type: str

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="after")
    def _check_ranges(self):
        pairs = (
            ("minimum_nights", "maximum_nights"),
            ("minimum_minimum_nights", "maximum_minimum_nights"),
            ("minimum_maximum_nights", "maximum_maximum_nights"),
            ("minimum_nights_avg_ntm", "maximum_nights_avg_ntm"),
        )
        for lower, upper in pairs:
            lower_val = getattr(self, lower)
            upper_val = getattr(self, upper)
            if lower_val is not None and upper_val is not None and lower_val > upper_val:
                raise ValueError(f"'{lower}' nao pode ser maior que '{upper}'.")

        total = self.host_total_listings_count
        count = self.host_listings_count
        if total is not None and count is not None and count > total:
            raise ValueError("host_listings_count nao pode ser maior que host_total_listings_count.")

        return self


# ----------------------------------------------------------
# Funcao de previsao
# ----------------------------------------------------------
@app.post("/predict")
def predict(data: AirbnbInput):
    """
    Faz a previsao da classe de preco (baixo, medio, luxo)
    e retorna tambem as explicacoes LIME e SHAP das principais variaveis.
    """
    payload = data.model_dump()
    _log_event("predict_request_received", payload=payload)
    try:
        df = _prepare_features(data)

        # Escalonar
        X_scaled = scaler.transform(df)

        # Previsao
        pred_num = model.predict(X_scaled)[0]
        classe_prevista = le.inverse_transform([pred_num])[0]
        prob = model.predict_proba(X_scaled)[0]
        probabilities_map = {c: float(p) for c, p in zip(le.classes_, prob)}

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

        # ---------- SHAP ----------
        try:
            if SHAP_BACKGROUND is None:
                shap_values_raw = SHAP_EXPLAINER.shap_values(X_scaled)
            else:
                shap_values_raw = SHAP_EXPLAINER.shap_values(X_scaled, nsamples="auto")

            shap_row = _select_shap_row(shap_values_raw, int(pred_num))
            explicacao_shap = build_shap_readable(
                shap_row,
                X_scaled[0],
                df,
                classe_prevista,
                topk=10,
                min_abs=0.0,
            )
        except Exception as shap_exc:
            explicacao_shap = {"erro": str(shap_exc)}
            _log_event("predict_shap_error", error=str(shap_exc))

        _log_event(
            "predict_response_ready",
            predicted_class=classe_prevista,
            confidence=probabilities_map.get(classe_prevista),
            probabilities=probabilities_map,
            lime_coverage=explicacao_legivel.get("cobertura_pct"),
            shap_coverage=explicacao_shap.get("cobertura_pct")
            if isinstance(explicacao_shap, dict)
            else None,
        )

        resultado = {
            "classe_prevista": classe_prevista,
            "confianca": f"{prob[pred_num]*100:.1f}%",
            "probabilidades": {
                c: f"{p*100:.1f}%" for c, p in probabilities_map.items()
            },
            "explicacao_LIME": explicacao_legivel,
            "explicacao_SHAP": explicacao_shap,
        }

        return {"status": "ok", "resultado": resultado}

    except HTTPException as exc:
        _log_event("predict_http_error", detail=exc.detail, payload=payload)
        raise exc
    except Exception as e:
        _log_event("predict_unexpected_error", error=str(e), payload=payload)
        logger.exception("predict_unexpected_error")
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
