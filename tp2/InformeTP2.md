| Preprocesamiento | Descripción | Nombre de función |
| ---------------- | ----------- | ----------------- |
| PP Arboles Decisión | Elimina columnas sin infromación valiosa, encodea variables categóricas y completa los missing values de la columna `edad`. | `procesamiento_arboles(dataframe)`|

*Tabla 1: Preprocesamientos.*

----
| Modelo | Preprocesamiento | AUC-ROC | Accuracy | Precision | Recall | F1 Score |
| ---------------- | ----------- | -------- | --------- | --------- | --------- | --------- |
| 1-Arbol de Decision | PP Arboles Decisión | 0.79 | 0.83 | 0.91 | 0.64 | 0.71 |

*Tabla 2: Modelos.*