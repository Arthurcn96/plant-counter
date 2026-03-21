# Abordagem Técnica

## Exploração dos dados

A primeira etapa de qualquer trabalho com dados novos é entender o que se tem em mãos. Por isso, antes de escrever qualquer linha de código de produção, desenvolvi um notebook de EDA (_Exploratory Data Analysis_) para inspecionar as imagens fornecidas.

Através do [rasterio](https://pypi.org/project/rasterio/), qualquer arquivo GeoTIFF carregado com `rasterio.open()` expõe atributos importantes como metadados do arquivo. Com isso foi possível extrair informações como o CRS (sistema de coordenadas), o GSD (_Ground Sampling Distance_) e as dimensões da imagem — dados essenciais para todas as etapas seguintes.

## Detecção das árvores

### Bandas RGB

Com experiência prévia em processamento de imagem, minha primeira abordagem foi analisar as bandas R, G e B separadamente. Para cada canal, apliquei o método de **Otsu** para encontrar automaticamente o threshold ideal entre as duas classes (planta e solo), seguido de **MORPH_OPEN** do OpenCV para remover ruídos da máscara binária resultante.

Com isso, obtemos 3 máscaras independentes — uma por canal — com regiões candidatas a árvores destacadas. Para detectar individualmente cada planta dentro dessas máscaras, avaliei duas abordagens:

- [Blob Detection](https://opencv.org/blob-detection-using-opencv/)
- [connectedComponentsWithStats](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f)

Optei pelo **connectedComponentsWithStats** por oferecer mais controle: permite filtrar por área mínima e máxima (descartando ruídos pequenos e fusões de copas), calcular a área de cada componente diretamente e tratar formas irregulares de forma mais flexível.

Ao aplicar a detecção nas 3 máscaras, obtemos 3 conjuntos de pontos independentes para a mesma imagem. A etapa seguinte consiste em comparar esses conjuntos e descartar detecções inconsistentes — um ponto só é considerado planta se aparecer em pelo menos 2 das 3 bandas simultaneamente. Essa abordagem de votação reduz falsos positivos causados por variações locais de iluminação ou reflexos que afetam apenas um canal de cor.

Vale mencionar que a detecção por bandas RGB tem limitações conhecidas — em particular, sofre com sombras intensas, como a da árvore adulta presente no `sample2.tif`. Essa limitação é parcialmente mitigada pela combinação com os índices de vegetação descritos a seguir.

### Índices de Vegetação

Uma abordagem complementar é utilizar índices de vegetação para destacar regiões com tonalidade esverdeada. Essa estratégia é especialmente útil em condições de iluminação variável, onde a análise por bandas isoladas pode ser insuficiente.

O ponto de partida foi o índice de marrom de Aimonino:

$$BI_{Aimonino} = kR - G - B \quad \text{com } k \in [0, 4]$$

Adaptei a fórmula para evidenciar o canal verde ao invés do vermelho, chegando ao índice:

$$EG_{Arthur} = kG - R - B \quad \text{com } k \in [0, 4]$$

Como a fórmula pode gerar valores negativos dependendo dos valores de R, G e B do pixel, é necessário normalizar o resultado para o intervalo [0, 255] antes de qualquer visualização ou binarização.

No total, foram avaliados 4 índices de vegetação, cada um com características diferentes de sensibilidade espectral:

|Índice|Fórmula|Característica|
|---|---|---|
|**ExG**|$4G - R - B$|Destaca o excesso de verde, base da abordagem|
|**Smolka**|$(G - \max(R,B)^2) / G$|Normalizado pelo canal verde, mais robusto a variações de brilho|
|**Vwg**|$(2G - (R+B)) / (2G + (R+B))$|Versão normalizada do ExG, valores entre -1 e 1|
|**LGI**|$-0.884R + 1.262G - 0.311B$|Pesos negativos para R e B, captura vegetação que os outros podem perder|

Para a binarização de cada índice, foram comparados três métodos:

- **Threshold manual** — valor fixo definido empiricamente a partir da análise do EDA
- **Otsu** — encontra automaticamente o threshold ideal baseado na distribuição do histograma
- **Triangle** — desenvolvido para imagens onde uma das classes é muito menor que a outra, adequado para cenários onde as copas ocupam uma fração pequena da imagem

O resultado final utiliza ExG e Smolka com Otsu e threshold manual combinando 4 máscaras independentes que alimentam a etapa de votação por consenso. Todas as outras não tiveram o resultado bom o suficiente, era possível chegar em resultados melhores caso mais tempo estivesse disponível, porém foi uma decisão estratégica.

### Combinação dos resultados

Após obter os pontos detectados tanto pelas bandas RGB quanto pelos índices de vegetação, os dois conjuntos são combinados através de um mecanismo de **votação por consenso**: pontos próximos (dentro de um raio configurável em pixels) detectados por fontes diferentes são agrupados e seu centroide é calculado. Apenas pontos confirmados por pelo menos 2 fontes independentes são mantidos no resultado final.

Essa estratégia aumenta a robustez da detecção sem depender de um único método, que pode falhar dependendo das condições de iluminação ou das características espectrais do solo.

## Limitações conhecidas

- **Sombras densas**: plantas sob sombra intensa têm brilho reduzido e podem ser perdidas pela máscara, especialmente na detecção por bandas RGB.
- **Solo esverdeado ou palhada**: solos com tonalidade verde ou presença de vegetação rasteira podem gerar falsos positivos nos índices de vegetação.
- **Copas sobrepostas**: plantas muito próximas podem ser fundidas em um único blob, sendo contadas como uma só.
- **Árvores adultas**: objetos muito maiores que o tamanho esperado de uma muda são filtrados pelo limite de área máxima, mas podem fragmentar e gerar múltiplas detecções se parcialmente sombreados.
- **Linhas de plantio**: o `sample2.tif` apresenta estruturas lineares no solo — restos de material orgânico ou sulcos — que conectam as plantas em diagonal, formando um padrão de grade. O `connectedComponentsWithStats` pode fundir plantas de uma mesma linha em um único blob, subestimando a contagem real.

## Melhorias futuras

Com mais tempo, as principais evoluções seriam:

- Utilizar a **banda NIR** (presente nos arquivos fornecidos) para calcular o **NDVI**, índice muito mais robusto que o ExG para separar vegetação de solo em qualquer condição de iluminação.
- Aplicar modelos de **segmentação semântica** (como U-Net) treinados com anotações das imagens, o que permitiria lidar com sombras e variações de solo de forma muito mais generalizada.
- Implementar **processamento em tiles** para suportar imagens de grande resolução sem limitações de memória.
- **Kernel direcional para quebrar linhas de plantio**: em vez do kernel elíptico padrão, aplicar erosões direcionais com kernels horizontais e verticais do tipo `MORPH_RECT` para destruir as estruturas lineares estreitas sem afetar as copas circulares. A ideia é que as linhas de tronco têm poucos pixels de largura — uma erosão com kernel `(15, 1)` destrói qualquer estrutura estreita nessa direção, preservando as copas que, por serem circulares e mais largas, sobrevivem à erosão. Uma dilatação elíptica posterior recupera o tamanho original das copas.
