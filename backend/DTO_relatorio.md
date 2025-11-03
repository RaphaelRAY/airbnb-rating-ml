# Relatorio do DTO aceito pelo endpoint `/predict`

Este documento descreve o contrato esperado pelo backend para realizar previsoes de classe de preco. O front deve enviar todos os campos listados abaixo no corpo JSON da requisicao.

## Estrutura geral

```json
{
  "...": "valores descritos nas tabelas abaixo"
}
```

Todos os numeros devem ser enviados como tipo numerico (sem aspas). Valores booleanos devem ser `true` ou `false`. Strings sao sensiveis ao conteudo informado, mas o backend realiza normalizacao simples (caixa alta/baixa, espacos, underscores) onde indicado.

## Campos numericos

| Campo | Tipo | Descricao | Observacoes |
| --- | --- | --- | --- |
| `host_response_rate` | float | Taxa de resposta do anfitriao (0.0 - 1.0). | Opcional validar para ficar entre 0 e 1. |
| `host_acceptance_rate` | float | Taxa de aceitacao do anfitriao (0.0 - 1.0). | Opcional validar para ficar entre 0 e 1. |
| `host_listings_count` | float | Qtde de anuncios atuais do anfitriao. | Aceita inteiros ou decimais. |
| `host_total_listings_count` | float | Qtde total de anuncios historicos do anfitriao. |  |
| `latitude` | float | Latitude do imovel. |  |
| `longitude` | float | Longitude do imovel. |  |
| `accommodates` | int | Quantidade de hospedes acomodados. |  |
| `bathrooms` | float | Numero de banheiros. |  |
| `bedrooms` | float | Numero de quartos. |  |
| `beds` | float | Numero de camas. |  |
| `minimum_nights` | int | Minimo de noites por reserva. |  |
| `maximum_nights` | int | Maximo de noites por reserva. |  |
| `minimum_minimum_nights` | float | Valor minimo historico do campo anterior. |  |
| `maximum_minimum_nights` | float | Valor maximo historico do campo `minimum_nights`. |  |
| `minimum_maximum_nights` | float | Valor minimo historico do campo `maximum_nights`. |  |
| `maximum_maximum_nights` | float | Valor maximo historico do campo `maximum_nights`. |  |
| `minimum_nights_avg_ntm` | float | Media de noites minimas (proximos 365 dias). |  |
| `maximum_nights_avg_ntm` | float | Media de noites maximas (proximos 365 dias). |  |
| `host_days_active` | int | Numero de dias que o anfitriao esta ativo. |  |
| `amenities_count` | int | Quantidade de comodidades listadas. |  |

O backend calcula internamente a feature derivada `bath_per_bedroom`; nao enviar esse campo.

## Campos booleanos

| Campo | Tipo | Descricao |
| --- | --- | --- |
| `host_has_profile_pic` | bool | Indica se o anfitriao possui foto de perfil. |
| `host_identity_verified` | bool | Indica se o anfitriao possui identidade verificada. |
| `has_availability` | bool | Indica disponibilidade futura do anúncio. |

Enviar `true` ou `false`; o backend converte para 1.0/0.0 automaticamente.

## Campos categoricos

| Campo | Tipo | Valores aceitos | Observacoes |
| --- | --- | --- | --- |
| `host_response_time` | string | `\"within an hour\"`, `\"within a few hours\"`, `\"within a day\"`, `\"a few days or more\"`, `\"unknown\"`. | Sao aceitas variantes como `WITHIN_A_DAY`, `Within a Day`, etc. Valores fora dessa lista retornam HTTP 422. |
| `room_type` | string | `\"Entire home/apt\"`, `\"Private room\"`, `\"Shared room\"`, `\"Hotel room\"`. | Variantes de caixa/espacos/sublinhados sao normalizadas. |
| `neighbourhood_cleansed` | string | Qualquer bairro presente no treinamento (ver `label_encoder_neighbourhood.pkl`). | Deve corresponder a um valor conhecido; caso contrario, HTTP 422. |
| `property_type` | string | Qualquer tipo de propriedade presente no treinamento (ver `label_encoder_property.pkl`). | Sensivel ao conteudo; caixa diferente e espacos adicionais sao ajustados. |

## Exemplo completo

```json
{
  "host_response_rate": 0.92,
  "host_acceptance_rate": 0.85,
  "host_listings_count": 3,
  "host_total_listings_count": 4,
  "host_has_profile_pic": true,
  "host_identity_verified": true,
  "latitude": -23.5587,
  "longitude": -46.6629,
  "accommodates": 4,
  "bathrooms": 1.5,
  "bedrooms": 2,
  "beds": 2,
  "minimum_nights": 2,
  "maximum_nights": 14,
  "minimum_minimum_nights": 1.0,
  "maximum_minimum_nights": 5.0,
  "minimum_maximum_nights": 30.0,
  "maximum_maximum_nights": 365.0,
  "minimum_nights_avg_ntm": 2.3,
  "maximum_nights_avg_ntm": 45.0,
  "has_availability": true,
  "host_days_active": 320,
  "amenities_count": 25,
  "host_response_time": "within a day",
  "room_type": "Entire home/apt",
  "neighbourhood_cleansed": "Pinheiros",
  "property_type": "Apartment"
}
```

## Resposta da API

Quando o payload é aceito, a API responde com HTTP 200:

```json
{
  "status": "ok",
  "resultado": {
    "classe_prevista": "medio",
    "confianca": "67.1%",
    "probabilidades": {
      "baixo": "12.4%",
      "medio": "67.1%",
      "luxo": "20.5%"
    },
    "explicacao_LIME": [
      ["bathrooms > 1.2", 0.145],
      ["amenities_count <= 20.0", -0.083],
      ["room_type_Private room <= 0.5", 0.064]
    ]
  }
}
```

- `classe_prevista`: classe final (`baixo`, `medio` ou `luxo`).
- `confianca`: probabilidade da classe prevista com uma casa decimal.
- `probabilidades`: mapa com todas as classes e suas probabilidades.
- `explicacao_LIME`: pares `[feature, impacto]` que mostram a contribuição local para a predição.

Se houver erro de validação (por exemplo, categoria não vista), o backend retorna HTTP 422 com mensagem detalhada. Falhas internas respondem com `{"status": "erro", "mensagem": "<detalhes>"}`.

Após o envio bem-sucedido, o backend gera automaticamente as features escalonadas, faz a predição e constrói a explicação LIME antes de montar a resposta.
