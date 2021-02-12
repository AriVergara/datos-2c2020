| Preprocesamiento | Descripción | Nombre de función |
| ---------------- | ----------- | ----------------- |
| PreprocessingLE | Elimina columnas sin infromación valiosa (`fila`, `id_usuario`, `id_ticket`), encodea variables categóricas mediante `LabelEncoding` (`genero`, `nombre_sala`, `tipo_de_sala`), completa los missing values de la columna `edad` y convierte en bins los valores de las columnas `edad` y `precio_ticket`. | `PreprocessingLE()`|

*Tabla 1: Preprocesamientos.*

----
| Modelo | Preprocesamiento | AUC-ROC | Accuracy | Precision | Recall | F1 Score |
| ---------------- | ----------- | -------- | --------- | --------- | --------- | --------- |
| 1-Arbol de Decision | PreprocessingLE | 0.82 | 0.85 | 0.89 | 0.69 | 0.78 |

*Tabla 2: Modelos.*