_DISCLAIMER: O Readme.md foi gerado por IA_
# 🌱 Plant Counter

O **Plant Counter** não é apenas um simples contador de pixels; ele funciona como um "comitê de especialistas" digital para o inventário florestal. Desenvolvido em Python, o sistema foi projetado para detectar e analisar mudas (como eucalipto) em imagens de alta resolução (TIFF) capturadas por drones.

### 🗳️ O Diferencial: O Sistema de Votação
Diferente de métodos tradicionais que confiam em apenas uma técnica, o Plant Counter utiliza um **Mecanismo de Consenso (Votação Espacial)**:
1.  **Múltiplos Olhares:** O programa analisa a imagem através de várias "lentes" diferentes (bandas RGB individuais e índices de vegetação como ExG e Smolka).
2.  **Debate Digital:** Cada técnica "vota" onde acredita estar uma planta.
3.  **Veredito:** Uma muda só é oficialmente contabilizada se receber o selo de aprovação de diferentes métodos simultaneamente. 

Isso reduz drasticamente os erros, ignorando distrações como sombras, restos de cultura ou variações de solo que poderiam enganar um contador comum.

<img width="1069" height="989" alt="image" src="https://github.com/user-attachments/assets/bfa040f8-a75b-4d1a-97db-05bdd369702f" />

---

## 🚀 Como Usar

### Pré-requisitos
- Python 3.13+.
- (Opcional) [uv](https://github.com/astral-sh/uv) para gerenciamento rápido de pacotes.

### Instalação e Execução

Você pode escolher entre usar o **uv** (recomendado) ou o **pip** tradicional.

#### Opção 1: Usando uv (Recomendado)
1. Instale as dependências:
   ```bash
   uv sync
   ```
2. Execute o pipeline:
   ```bash
   uv run main.py
   ```

#### Opção 2: Usando pip tradicional
1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate  # Windows
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o pipeline:
   ```bash
   python main.py
   ```

---

### Preparação dos Dados
1. Coloque sua imagem TIFF na pasta `data/`.
2. Ajuste os parâmetros no arquivo `config.yaml` (caminhos, limiares de área, etc.).

**Após a execução, o output é gerado (ex: output/exp#/)**
---

## 📂 Estrutura de Pastas

- **`data/`**: Armazena as imagens de entrada (TIFF) e arquivos auxiliares.
- **`src/`**: Contém o código-fonte modular do sistema.
- **`notebooks/`**: Ambientes Jupyter para experimentação e Análise Exploratória de Dados (EDA).
- **`output/`**: Resultados gerados por cada execução (organizados por experimentos `exp1`, `exp2`, etc.).
- **`tests/`**: Local destinado a testes unitários (atualmente em desenvolvimento).

---

## 📊 O que sai na pasta de Output?

Para cada execução, uma subpasta (ex: `output/exp1/`) é criada contendo:

### Imagens (`imagens/`)
- **`original.jpg`**: Cópia da imagem original em formato JPG.
- **`preprocessed.jpg`**: Imagem após normalização e CLAHE.
- **`plot_bands.png`**: Grid mostrando a segmentação individual de cada banda RGB.
- **`plot_green_indices.png`**: Grid mostrando a segmentação por índices ExG e Smolka.
- **`final_detections_rgb.jpg` / `final_detections_green.jpg`**: Visualização dos pontos confirmados (ciano) e rejeitados (vermelho) sobrepostos à imagem.

### Dados Geoespaciais
- **`plantas.geojson`**: Pontos georreferenciados de todas as plantas confirmadas.
- **`plantas_poligonos.geojson`**: Polígonos georreferenciados representando a copa de cada planta, incluindo o cálculo de área em m².

### Relatórios e Metadados
- **`stats.json`**: Resumo estatístico (Total de plantas, plantas/ha, área total, homogeneidade e CV%).
- **`metadata.json`**: Metadados técnicos da imagem original (GSD, CRS, Dimensões).
- **`config_used.json`**: Cópia dos parâmetros utilizados naquela execução específica.
- **`analysis.png`**: Dashboard final contendo:
  1. Histograma de distribuição das áreas das copas.
  2. Mapa de calor (densidade relativa) das plantas.
  3. Histograma de espaçamento entre plantas (vizinho mais próximo).
