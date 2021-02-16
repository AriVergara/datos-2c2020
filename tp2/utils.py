# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import preprocesing as pp
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import preprocesing as pp


def kfold_for_cross_validation():
    return StratifiedKFold(n_splits=8, random_state=pp.RANDOM_STATE, shuffle=True)


def metricas_finales(y_test, y_pred, y_pred_proba):
    scores = [accuracy_score, precision_score, recall_score, f1_score]
    columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    results = [roc_auc_score(y_test, y_pred_proba)]
    results += [s(y_test, y_pred) for s in scores]
    display(pd.DataFrame([results], columns=columnas).style.hide_index())
    return results


def metricas_cross_validation_con_cross_validate(X, y, classifier):
    cv = kfold_for_cross_validation()
    scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    scores_for_model = cross_validate(classifier, X, y, cv=cv, scoring=scoring_metrics)
    print(f"Mean test roc auc is: {scores_for_model['test_roc_auc'].mean():.4f}, standard deviation is: {scores_for_model['test_roc_auc'].std():.4f}")
    print(f"mean test accuracy is: {scores_for_model['test_accuracy'].mean():.4f}, standard deviation is: {scores_for_model['test_accuracy'].std():.4f}")
    print(f"mean test precision is: {scores_for_model['test_precision'].mean():.4f}, standard deviation is: {scores_for_model['test_precision'].std():.4f}")
    print(f"mean test recall is: {scores_for_model['test_recall'].mean():.4f}, standard deviation is: {scores_for_model['test_recall'].std():.4f}")
    print(f"mean test f1_score is: {scores_for_model['test_f1'].mean():.4f}, standard deviation is: {scores_for_model['test_f1'].std():.4f}")
    return scores_for_model


def metricas_cross_validation(X, y, clf):
    kf = kfold_for_cross_validation()

    test_accuracies = []
    test_roc_aucs = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    oof_predictions = np.zeros(len(X))
    oof_predictions_proba = np.zeros(len(X))
    
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        y_test_cv = y[test_index]
        
        clf.fit(X.loc[train_index,], y[train_index])
        y_predict_cv = clf.predict(X.loc[test_index,])
        y_predict_proba_cv = clf.predict_proba(X.loc[test_index,])[:, 1]
        
        oof_predictions[test_index] = y_predict_cv
        oof_predictions_proba[test_index] = y_predict_proba_cv

        test_roc_auc = roc_auc_score(y_test_cv, y_predict_proba_cv)
        test_roc_aucs.append(test_roc_auc)

        test_accuracy = accuracy_score(y_test_cv, y_predict_cv)
        test_accuracies.append(test_accuracy)

        test_precision = precision_score(y_test_cv, y_predict_cv)
        test_precisions.append(test_precision)

        test_recall = recall_score(y_test_cv, y_predict_cv)
        test_recalls.append(test_recall)

        test_f1_score = f1_score(y_test_cv, y_predict_cv)
        test_f1_scores.append(test_f1_score)

    data = [
        [np.mean(test_roc_aucs), np.std(test_roc_aucs), roc_auc_score(y, oof_predictions_proba), np.max(test_roc_aucs), np.min(test_roc_aucs)],
        [np.mean(test_accuracies), np.std(test_accuracies), accuracy_score(y, oof_predictions), np.max(test_accuracies), np.min(test_accuracies)],
        [np.mean(test_precisions), np.std(test_precisions), precision_score(y, oof_predictions), np.max(test_precisions), np.min(test_precisions)],
        [np.mean(test_recalls), np.std(test_recalls), recall_score(y, oof_predictions), np.max(test_recalls), np.min(test_recalls)],
        [np.mean(test_f1_scores), np.std(test_f1_scores), f1_score(y, oof_predictions), np.max(test_f1_scores), np.min(test_f1_scores)]
    ]
    columns = ["Mean", "Std", "Oof", "Max", "Min"]
    index = ["roc auc", "accuracy", "precision", "recall", "f1 score"]
    display(pd.DataFrame(data, columns=columns, index=index))
    #print(f"mean test roc auc is: {np.mean(test_roc_aucs):.4f}, standard deviation is: {np.std(test_roc_aucs):.4f}, oof is: {roc_auc_score(y, oof_predictions_proba)}")
    #print(f"mean test accuracy is: {np.mean(test_accuracies):.4f}, standard deviation is: {np.std(test_accuracies):.4f}, oof is {accuracy_score(y, oof_predictions)}")
    #print(f"mean test precision is: {np.mean(test_precisions):.4f}, standard deviation is: {np.std(test_precisions):.4f}, oof is {precision_score(y, oof_predictions)}")
    #print(f"mean test recall is: {np.mean(test_recalls):.4f}, standard deviation is: {np.std(test_recalls):.4f}, oof is {recall_score(y, oof_predictions)}")
    #print(f"mean test f1_score is: {np.mean(test_f1_scores):.4f}, standard deviation is: {np.std(test_f1_scores):.4f}, oof is {f1_score(y, oof_predictions)}")


def entrenar_y_realizar_prediccion_final_con_metricas(X, y, pipeline, use_decision_function=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=pp.RANDOM_STATE, stratify=y)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if use_decision_function:
        y_pred_proba = pipeline.decision_function(X_test)
    else:
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
    scores = [accuracy_score, precision_score, recall_score, f1_score]
    columnas = ['AUC_ROC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    results = [roc_auc_score(y_test, y_pred_proba)]
    results += [s(y_test, y_pred) for s in scores]
    display(pd.DataFrame([results], columns=columnas).style.hide_index())
    return pipeline


def predecir_holdout_y_generar_csv(pipeline, path_archivo):
    df_predecir = pd.read_csv('https://drive.google.com/uc?export=download&id=1I980-_K9iOucJO26SG5_M8RELOQ5VB6A')
    df_predecir['volveria'] = pipeline.predict(df_predecir)
    df_predecir = df_predecir[['id_usuario', 'volveria']]
    with open(path_archivo, 'w') as f:
        df_predecir.to_csv(f, sep=',', index=False)


def importar_datos():
    df_volvera = pd.read_csv('tp-2020-2c-train-cols1.csv')
    df_volvera.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
    df_datos = pd.read_csv('tp-2020-2c-train-cols2.csv')
    df_datos.rename(columns={c: c.lower().replace(" ","_") for c in df_volvera.columns}, inplace=True)
    df = df_volvera.merge(df_datos, how='inner', right_on='id_usuario', left_on='id_usuario')
    X = df.drop(columns="volveria", axis=1, inplace=False)
    y = df["volveria"]
    return X, y
