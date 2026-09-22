# text4VAR
## Resultados de experimentos con UCF-Crime:
### Experimento: CLIP (ViT-L/14) con matriz de clasificación de centroides (calculados por *promedio*) de descripciones de segmentos anómalos
1. Descripción: sin **Dropout**, con **Label Smoothing**, sin **WeightedRandomSampler**
  * Parámetros:
    * Dropout: 0.0
    * Loss Function: Cross Entropy
      * Labels Smoothing: 0.1
    * Sampler: N/A
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 53.21%, Top@3: 76.92%
      * Precision: 48.54
      * Recall: 42.25
      * F1-score: 40.51
    * A nivel de video:
      * Soft Voting: Top@1: 53.57%, Top@3: 77.14%
      * Hard Voting: Top@1: 54.29%, Top@3: 67.86%
2. Descripción: sin **Dropout**, con **Label Smoothing**, sin **WeightedRandomSampler**
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
3. Descripción: con **Dropout**, con **Label Smoothing**, sin **WeightedRandomSampler**
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
4. Descripción: sin **Dropout**, con **Label Smoothing**, con **WeightedRandomSampler**
  * Parámetros: 
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
5. Descripción: con **Dropout**, con **Label Smoothing**, con **WeightedRandomSampler**
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
### Experimento: CLIP (ViT-L/14) con matriz de clasificación de centroides (calculados por *promedio*) de descripciones de segmentos anómalos y con reducciones de dimensionalidad en esta y en el embedding de salida del transformer temporal. Todos estos experimentos son sin usar dropout, label smoothing y WeightedRandomSampler
6. Descripción: Reduciendo con $\text{PCA}_{\omega=0.75}, d=562$
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 53.21%, Top@3: 74.36%
      * Precision: 44.94
      * Recall: 46.73
      * F1-score: 42.24
    * A nivel de video:
      * Soft Voting: Top@1: 54.29%, Top@3: 75.71%
      * Hard Voting: Top@1: 54.29%, Top@3: 70.00%
7. Descripción: Reduciendo con $\text{PCA}_{\omega=0.50}, d=384$
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 51.92%, Top@3: 77.56%
      * Precision: 51.17
      * Recall: 45.95
      * F1-score: 42.34
    * A nivel de video:
      * Soft Voting: Top@1: 53.57%, Top@3: 77.14%
      * Hard Voting: Top@1: 53.57%, Top@3: 74.29%
8. Descripción: Reduciendo con $\text{PCA}_{\omega=0.25}, d=192$
  * Resultados:
    * A nivel de segmento:
      * Accuracy: Top@1: 53.21%, Top@3: 75.00%
      * Precision: 43.21
      * Recall: 41.84
      * F1-score: 39.15
    * A nivel de video:
      * Soft Voting: Top@1: 54.29%, Top@3: 75.71%
      * Hard Voting: Top@1: 54.29%, Top@3: 68.57%
### Experimento: CLIP (ViT-L/14) con matriz de clasificación de descripciones sintéticas de las anomalías y con reducciones de dimensionalidad en esta y en el embedding de salida del transformer temporal. Todos estos experimentos son sin usar dropout, label smoothing y WeightedRandomSampler
9. Resultados:
  * A nivel de segmento:
    * Accuracy: Top@1: 51.92%, Top@3: 80.13%
    * Precision: 48.78
    * Recall: 39.69
    * F1-score: 39.05
  * A nivel de video:
    * Soft Voting: Top@1: 52.14%, Top@3: 80.00%
    * Hard Voting: Top@1: 52.14%, Top@3: 67.86%
