# Airbnb Rating ML

## Visão Geral

Este projeto busca prever a **nota média (`review_scores_rating`) de imóveis novos no Airbnb** utilizando técnicas de aprendizado de máquina. A ideia é ajudar a estimar a reputação esperada de um imóvel com base em características como localização, tipo, preço e informações do anfitrião.

---

## Objetivo

- Prever a nota esperada de novos anúncios sem avaliações.
- Fornecer uma base para análise de fatores que influenciam a reputação.

---

## Tecnologias

- **Python**
- **XGBoost**
- **Scikit-learn**
- **SHAP / LIME** (para interpretação do modelo)

---

## Estrutura Simplificada

```
airbnb-rating-ml/
├── data/         # Dados brutos e tratados
├── notebooks/    # Análises e experimentos
├── backend/      # API e ML
├── frontend/     # interface do usuário
└── models/       # Modelos treinados
```

## Autor

## **Raphael Raymundo**

---

> _Modelos preditivos são úteis quando também são compreensíveis._
## Execucao com Docker Compose

1. Certifique-se de ter **Docker** e **Docker Compose** instalados.
2. Na raiz do projeto, execute:

   ```bash
   docker compose up --build
   ```

3. Os servicos sobem com as seguintes portas expostas:
   - Backend FastAPI em `http://localhost:8000`
   - Frontend Next.js em `http://localhost:3000`

O volume `./models` e montado no container para garantir o acesso aos artefatos do modelo. O frontend recebe automaticamente a variavel `NEXT_PUBLIC_API_URL` apontando para o backend, entao a aplicacao ja inicia integrada.
