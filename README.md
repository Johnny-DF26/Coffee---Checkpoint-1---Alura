# 🌟 VitalityTech: Análise Preditiva da Qualidade do Sono e Nível de Estresse

## 📝 Visão Geral do Projeto
Este projeto visa investigar a relação entre hábitos de vida, como consumo de café, horas de sono, atividade física e dados demográficos, com os níveis de estresse e a qualidade do sono, especificamente para a empresa VitalityTech. O objetivo é desmistificar crenças internas (como a do café como principal causador de estresse) e desenvolver um modelo preditivo para identificar indivíduos em risco de alto estresse.

## 🎯 Objetivos Principais
1.  **Identificar Padrões**: Analisar o impacto do consumo de cafeína na frequência cardíaca e no estresse.
2.  **Mapear o Sono**: Quantificar a relação entre privação de sono e níveis de estresse.
3.  **Modelo Preditivo**: Desenvolver um classificador para prever usuários com 'Alto Estresse' e permitir intervenções personalizadas.

## 📊 Fonte dos Dados
Os dados sintéticos utilizados foram obtidos do Kaggle: [Global Coffee Health Dataset](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset).

## 🛠️ Metodologia

### 1. Importação e Configuração
-   Carregamento das bibliotecas essenciais (`pandas`, `matplotlib`, `seaborn`, `sklearn`, `imblearn`).
-   Configuração de opções de exibição do Pandas para melhor visualização.

### 2. Base de Dados
-   Leitura do arquivo `synthetic_coffee_health_10000(in).csv`.
-   Tradução das colunas para português para maior clareza.
-   Visualização inicial do DataFrame (`df.head()`, `df.shape`).

### 3. Análise Exploratória de Dados (EDA)
-   **Análise Estatística**: Uso de `df.info()` e `df.describe()` para entender a estrutura e estatísticas básicas dos dados.
-   **Análise Univariada**: Visualização da distribuição de `idade`, `gênero`, `país`, `consumo_diario_cafe`, `cafeina_ingerida_dia`, `horas_de_sono`, `qualidade_do_sono`, `imc`, `frequencia_cardiaca`, `nivel_estresse`, `atividade_fisica`, `condicoes_saude`, `ocupacao`, `fumante` e `consume_alcool` por meio de histogramas, box-plots e gráficos de contagem, com insights detalhados para cada variável.
-   **Análise Bivariada e Correlações**: 
    -   **Matriz de Correlações**: Mapa de calor para identificar relações entre variáveis numéricas, revelando alta multicolinearidade entre `consumo_diario_cafe` e `cafeina_ingerida_dia`.
    -   **Gênero x Qualidade do Sono**: Análise da distribuição da qualidade do sono por gênero.
    -   **Consumo de Café x Frequência Cardíaca**: Scatter plot com regressão, mostrando baixa correlação.
    -   **Consumo de Café x Horas de Sono**: Scatter plot com regressão, indicando uma leve correlação negativa.
    -   **Consumo de Café x Fumante**: Comparação por meio de histogramas e box-plots, desmistificando a correlação comum.
    -   **Consumo de Café x Ocupação e Idade**: Análise da homogeneidade do consumo de café entre diferentes grupos.
    -   **Países que mais consomem café**: Ranking e análise da uniformidade do consumo global.
-   **Análise com a Variável Alvo (`nivel_estresse`)**:
    -   **Nível de Estresse x Gênero, Qualidade do Sono, Horas de Sono, Idade, Países, Ocupações e Frequência Cardíaca**: Visualizações e insights sobre a interação dessas variáveis com o nível de estresse.

### 4. Data Cleaning
-   Criação de uma cópia do DataFrame (`df_clean`).
-   Tratamento de valores nulos na coluna `condicoes_saude` (removida devido à alta proporção de NaNs e baixa relevância preditiva).
-   Verificação de dados duplicados (nenhum encontrado).
-   Remoção de colunas irrelevantes (`id`).
-   Tratamento de redundância: remoção de `consumo_diario_cafe` devido à alta correlação com `cafeina_ingerida_dia`.

### 5. Feature Engineering
-   Criação de `categoria_imc` (Abaixo do peso, Peso normal, Sobrepeso, Obesidade).
-   Criação de `faixa_etaria` (Jovem, Adulto, Sênior, Idoso).
-   Conversão do tipo de dados de `faixa_etaria` para `object`.

### 6. Pré-processamento para Machine Learning
-   **Divisão em Treino e Teste**: Separação do DataFrame em `X` (features) e `y` (target: `qualidade_do_sono`).
-   **One-Hot Encoding e Padronização**: Utilização de `ColumnTransformer` para aplicar `OneHotEncoder` em colunas categóricas e `StandardScaler` em numéricas.
-   **Label Encoding**: Transformação da variável alvo `y` para representação numérica.

### 7. Feature Selection
-   **Análise de Multicolinearidade (VIF)**: Avaliação do Variance Inflation Factor (VIF) para identificar e remover features altamente correlacionadas, como `nivel_estresse_Low`, que surgem da codificação One-Hot.

### 8. Machine Learning (Seleção de Modelo)
-   **Modelos Testados**:
    -   **Naive Bayes (GaussianNB)**: Implementação e avaliação da acurácia e relatório de classificação.
    -   **Árvore de Decisão (DecisionTreeClassifier)**: Implementação e avaliação, com ajuste de `max_depth`.
    -   **Random Forest (RandomForestClassifier)**: Implementação e avaliação, com ajuste de `n_estimators` e `max_depth`.
-   **Importância das Variáveis**: Análise das features mais importantes para o modelo Random Forest, destacando `horas_de_sono` e `stress_score`.
-   **Balanceamento dos Dados (SMOTE)**: Aplicação do SMOTE para lidar com o desbalanceamento das classes da variável alvo, treinando o Random Forest em dados resampleados para melhorar o recall das classes minoritárias.
-   **Validação Cruzada**: Uso de `StratifiedKFold` para avaliar a robustez dos modelos.

### 9. Modelo Final
-   O modelo `RandomForestClassifier` (com `n_estimators=100`, `max_depth=6`, `random_state=42`) foi escolhido e treinado no conjunto completo de dados pré-processados (sem as remoções experimentais de features principais).
-   O pipeline completo (pré-processamento + modelo) foi salvo utilizando `joblib` para futuras previsões.

### 10. Prevendo Novos Valores
-   Carregamento do modelo salvo.
-   Demonstração de como fazer previsões para novos dados de entrada, traduzindo o resultado numérico da qualidade do sono para categorias legíveis.

## 💡 Insights Conclusivos

-   O **sono** (horas e qualidade) é o fator mais crítico e diretamente correlacionado ao **nível de estresse**. Indivíduos com estresse elevado tendem a ter menos horas de sono e pior qualidade.
-   O **consumo de café e cafeína** mostraram **pouca ou nenhuma correlação direta** com a frequência cardíaca ou o nível de estresse, desmistificando a crença popular.
-   Variáveis demográficas como **idade, gênero, país e ocupação** não se mostraram preditores fortes isolados do estresse ou consumo de café, sendo o comportamento de consumo homogêneo entre a maioria dos grupos.
-   O **modelo Random Forest** atingiu alta acurácia (próxima de 99%) ao prever a qualidade do sono, mas a análise de importância de features revelou que grande parte do poder preditivo reside em `horas_de_sono` e `nivel_estresse`.
-   A alta performance pode ser influenciada pela natureza sintética dos dados, onde as relações podem ser mais determinísticas do que em cenários reais. Recomenda-se cautela na interpretação e validação com dados do mundo real.

### ✅ Próximos Passos
-   Validar o modelo e os insights com dados reais.
-   Explorar modelos mais complexos ou técnicas de ensemble para verificar se capturam nuances não lineares entre as variáveis.
-   Desenvolver estratégias de intervenção personalizadas para a VitalityTech com base nos insights sobre sono e estresse.