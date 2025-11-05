# Classificação de Preços de Anúncios Airbnb no Rio de Janeiro

Este projeto constrói um sistema de **classificação de faixas de preço** para anúncios do Airbnb na cidade do Rio de Janeiro. A aplicação final recebe informações resumidas de um imóvel (número de quartos, tipo de quarto, localização, etc.) e retorna se o anúncio pertence à faixa **baixa**, **média** ou **luxo**. O trabalho foi desenvolvido em várias etapas — exploração dos dados, limpeza e pré-processamento, modelagem e avaliação — e culminou em uma API pronta para ser consumida por aplicações web ou serviços internos.

## Visão geral do conjunto de dados

Os dados provêm de um repositório público de listagens Airbnb. O conjunto principal (`listings.csv`) possui **79 colunas** e **42 572 registros**, com características que variam de descrições textuais até valores numéricos como preço, número de banheiros e latitude/longitude. Durante a exploração verificou-se que a base é bastante desequilibrada geograficamente: bairros turísticos concentram a maior parte dos anúncios (por exemplo, Copacabana possui mais de **13 000 anúncios**, seguida de Barra da Tijuca com ~**3 680** e Ipanema com ~**3 615**).
A maioria dos imóveis é do tipo “**Entire home/apt**” (cerca de **34 000 listagens**), enquanto quartos privados são pouco mais de 8 000 e quartos de hotel são raríssimos.

O arquivo `calendar.csv` traz informações de disponibilidade por data e confirmou que a maioria dos imóveis fica livre durante grande parte do ano, com variação sazonal nas datas de alta temporada. Já `neighbourhoods.csv` e `neighbourhoods.geojson` fornecem a estrutura administrativa dos bairros, permitindo análises espaciais e mapeamento em geovisualizações.

## Análise exploratória

Na etapa exploratória foram geradas estatísticas descritivas, perfis de anfitriões e imóveis e gráficos da distribuição de preços. Observou-se que:

* **Distribuição de preços:** assimetria positiva, com muitos anúncios baratos e poucos extremamente caros. Foi usada transformação logarítmica.
* **Valores ausentes:** removidos ou imputados.
* **Correlação entre atributos:** número de quartos, tipo de quarto e localização têm forte influência no preço.

## Limpeza e pré-processamento

* Remoção de colunas irrelevantes ou altamente correlacionadas
* Conversão de colunas categóricas em variáveis dummy
* Normalização e padronização de atributos numéricos
* Imputação de valores ausentes
* Detecção e remoção de outliers

Após o pré-processamento, o conjunto foi dividido em treino e teste (80/20). Também foi criada a variável-alvo categórica (`faixa_preco`) com três classes: **baixo**, **médio** e **luxo**.

## Modelagem e resultados

Três algoritmos foram avaliados:

| Modelo              | Acurácia   | Observação              |
| ------------------- | ---------- | ----------------------- |
| Regressão Logística | 0.5901     | Base de comparação      |
| Random Forest       | 0.6689     | Bom equilíbrio geral    |
| **XGBoost**         | **0.6917** | Melhor desempenho geral |

Além da acurácia, foram analisadas métricas como **Precisão**, **Recall** e **F1-score**, com bom desempenho nas três classes.

## Explicabilidade e API

* **LIME (Local Interpretable Model-agnostic Explanations):** explicações locais para previsões individuais.
* **SHAP (SHapley Additive Explanations):** identifica as variáveis que mais influenciam o modelo globalmente.
  * SHAP não funcional devido a problema de compatibilidade com o modelo XGBoost

A API, construída com **FastAPI**, carrega o modelo e os objetos de pré-processamento e expõe endpoints REST para previsões e explicações. Inclui CORS, logging estruturado e tratamento de erros.


## Conclusão

 O projeto demonstrou que, com pré‑processamento adequado e modelos de classificação robustos, é possível prever a faixa de preço de anúncios do Airbnb no Rio de Janeiro com acurácia superior a 69 %.O modelo XGBoost se destacou por capturar relações não lineares e minimizar a influência de outliers. A integração de ferramentas de interpretabilidade (LIME e SHAP) tornou o modelo transparente, permitindo identificar os fatores que mais influenciam o preço e fornecer confiança aos usuários e aos anfitriões.

 Como trabalhos futuros, sugerem‑se a inclusão de variáveis externas (como eventos sazonais e clima) e a experimentação com métodos de redes neurais. A API desenvolvida pode ser integrada a painéis web ou aplicativos móveis para ajudar anfitriões a precificar seus imóveis de forma competitiva e justa
 
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

## Execução com Docker Compose

1. Certifique-se de ter **Docker** e **Docker Compose** instalados.

2. Na raiz do projeto, execute:

   ```bash
   docker compose up --build
   ```

3. Os serviços sobem com as seguintes portas expostas:

   * Backend FastAPI → `http://localhost:8000`
   * Frontend Next.js → `http://localhost:3000`

O volume `./models` é montado no container para garantir acesso aos artefatos do modelo. O frontend recebe automaticamente `NEXT_PUBLIC_API_URL` apontando para o backend, iniciando o sistema integrado.

---

## Autores

*  **Raphael Raymundo** 21.00334-3
*  **Nicole Martins Fragnan** 21.00368-8
*  **Kaiven Yang Su** 20.02146-0
*  **Felipe Rodrigues Peixoto da Silva** 21.00127-8
*  **Eduardo Lucas Felippa** 20.01913-0
*  **Flávio Murata** 21.01192-3

## Referências

* Rogerio‑mack. IMT_CD_2025 – repositório GitHub de referência. Disponível em: https://github.com/Rogerio-mack/IMT_CD_2025
. Acesso em nov. 2025.
* LIME: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD.
* SHAP: Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.
* Airbnb data: dados públicos do portal Inside Airbnb. Disponíveis em insideairbnb.com (acesso em nov. 2025).

