# text4VAR
### Resultados de experimentos con UCF-Crime:
Experimento: CLIP (ViT-L/14) con matriz de clasificación de centroides de descripciones de video segmentos anómalos
* Descripción: Sin **Dropout** y con **Label Smoothing**
  * Parámetros:
    * Dropout: 0.0
    * Loss Function: Cross Entropy
      * Labels Smoothing: 0.1
    * Sampler: N/A
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 53.21%, Top@3: 76.28%
      * Precision: 52.33
      * Recall: 46.47
      * F1-score: 42.39
    * A nivel de video:
      * Soft Voting: Top@1: 54.29, Top@3: 77.14
      * Hard Voting: Top@1: 53.57, Top@3: 72.14
* Descripción: Con **Dropout** y con **Label Smoothing**
  * Parámetros:
    * Dropout: 0.3
    * Loss Function: Cross Entropy
      * Labels Smoothing: 0.1
    * Sampler: N/A
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 53.21%, Top@3: 73.72%
      * Precision: 43.66
      * Recall: 41.28
      * F1-score: 38.52
    * A nivel de video:
      * Soft Voting: Top@1: 54.29%, Top@3: 75.00%
      * Hard Voting: Top@1: 54.29%, Top@3: 75.00%
* Descripción: Con **Dropout**, **Label Smoothing** y con **WeightedRandomSampler**
  * Parámetros:
    * Dropout: 0.3
    * Loss Function: Cross Entropy
      * Labels Smoothing: 0.1
    * Sampler: WeightedRandomSampler
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 55.13%, Top@3: 80.13%
      * Precision: 54.73
      * Recall: 47.02
      * F1-score: 44.84
    * A nivel de video:
      * Soft Voting: Top@1: 57.14%, Top@3: 80.00%
      * Hard Voting: Top@1: 57.14%, Top@3: 69.29%
* Descripción: 
  * Parámetros: Sin **Dropout**, con **Label Smoothing** y **WeightedRandomSampler**
    * Dropout: 0.0
    * Loss Function: Cross Entropy
      * Labels Smoothing: 0.1
    * Sampler: WeightedRandomSampler
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 51.92%, Top@3: 72.44%
      * Precision: 49.92
      * Recall: 42.08
      * F1-score: 38.19
    * A nivel de video:
      * Soft Voting: Top@1: 57.14%, Top@3: 80.00%
      * Hard Voting: Top@1: 57.14%, Top@3: 69.29%
