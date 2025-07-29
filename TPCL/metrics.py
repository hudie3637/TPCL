import numpy as np
from sklearn.metrics import (
    accuracy_score as _accuracy_score,
    precision_score as _precision_score,
    recall_score as _recall_score,
    f1_score as _f1_score,
    roc_auc_score as _roc_auc_score,
    average_precision_score as _average_precision_score,
    confusion_matrix as _confusion_matrix,
    classification_report as _classification_report,
    roc_curve as _roc_curve,
    precision_recall_curve as _precision_recall_curve
)

def accuracy_score(y_true, y_pred):
    """计算准确率"""
    return _accuracy_score(y_true, y_pred)

def precision_score(y_true, y_pred, average='binary', pos_label=1):
    """计算精确率"""
    return _precision_score(y_true, y_pred, average=average, pos_label=pos_label)

def recall_score(y_true, y_pred, average='binary', pos_label=1):
    """计算召回率"""
    return _recall_score(y_true, y_pred, average=average, pos_label=pos_label)

def f1_score(y_true, y_pred, average='binary', pos_label=1):
    """计算F1分数"""
    return _f1_score(y_true, y_pred, average=average, pos_label=pos_label)

def roc_auc_score(y_true, y_score, average='macro', multi_class='ovr'):
    """计算ROC曲线下面积(AUC)"""
    return _roc_auc_score(y_true, y_score, average=average, multi_class=multi_class)

def average_precision_score(y_true, y_score, average='macro'):
    """计算平均精确率"""
    return _average_precision_score(y_true, y_score, average=average)

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    """计算混淆矩阵"""
    return _confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)

def classification_report(y_true, y_pred, target_names=None, sample_weight=None):
    """生成分类报告"""
    return _classification_report(y_true, y_pred, target_names=target_names, sample_weight=sample_weight)

def roc_curve(y_true, y_score, pos_label=None):
    """计算ROC曲线点"""
    return _roc_curve(y_true, y_score, pos_label=pos_label)

def precision_recall_curve(y_true, probas_pred, pos_label=None):
    """计算精确率-召回率曲线点"""
    return _precision_recall_curve(y_true, probas_pred, pos_label=pos_label)    