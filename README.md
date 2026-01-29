# EDA Master: Insights Estatísticos e Visualização de Dados com Python

Este repositório contém um laboratório prático de **Análise Exploratória de Dados (EDA)**, focado em transformar dados brutos em inteligência visual e métricas estatísticas acuradas. O projeto exercita o ciclo completo de tratamento de dados, desde a normalização de variáveis até a interpretação de distribuições complexas.

## 🚀 Competências Demonstradas

* **Engenharia de Atributos (Feature Engineering):** Transformação de unidades de medida (Hectares para Km²) para melhorar a interpretabilidade dos dados.
* **Análise Estatística Descritiva:** Cálculo e interpretação de tendência central (Média/Mediana) e dispersão (Desvio Padrão) para identificação de *outliers*.
* **Data Storytelling:** Construção de visualizações otimizadas com `Seaborn` e `Matplotlib` para traduzir distribuições numéricas e frequências de categorias.

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **Pandas:** Manipulação e limpeza de dados.
* **NumPy:** Suporte matemático para operações vetoriais.
* **Matplotlib & Seaborn:** Visualização de dados avançada e estilização de gráficos.

## 📊 O que este projeto analisa?

A partir de um dataset geográfico (BC250/IBGE), o script extrai:

1. **Distribuição Territorial:** Um histograma com estimativa de densidade (KDE) para entender a concentração das áreas.
2. **Status Jurídico:** Um ranking de frequência para identificar gargalos em processos administrativos/legais.
3. **Sumarização Crítica:** Um relatório de métricas que revela a heterogeneidade do conjunto de dados.

## 📖 Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/MariaSan-tos/Exploratory-Data-Analysis---training-1-.git

```


2. Certifique-se de ter as bibliotecas instaladas:
```bash
pip install pandas numpy matplotlib seaborn

```


3. Execute o script principal ou o notebook no ambiente de sua preferência (VS Code, Jupyter, Google Colab).

---

### 💡 Insight de Valor

Ao analisar a relação entre a **Média** e a **Mediana** geradas pelo código, é possível identificar imediatamente se o dataset possui uma distribuição assimétrica, competência essencial para validar qualquer modelo de Machine Learning ou relatório de Business Intelligence.
