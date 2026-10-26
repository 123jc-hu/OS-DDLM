import torch
import torchmetrics

def cal_auc(y_true, y_score, multi_class='raise'):
    """
    计算多分类/二分类AUC
    假设y_score是模型输出的概率（shape: [n_samples, n_classes]）
    """
    if multi_class == 'raise':
        multi_class = 'ovr' if y_score.shape[1] > 2 else 'binary'
    
    task_type = 'multiclass' if multi_class == 'ovr' else 'binary'
    num_classes = y_score.shape[1] if task_type == 'multiclass' else None
    
    auroc = torchmetrics.AUROC(
        task=task_type,
        num_classes=num_classes,
        average='macro' if task_type == 'multiclass' else None
    )
    return auroc(y_score, y_true)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算平衡准确率、混淆矩阵、TPR/TNR，以及AUC（若提供y_prob）
    
    参数：
        y_true: 真实标签（tensor, shape=[N]）
        y_pred: 预测标签（tensor, shape=[N]）
        y_prob: 预测概率或logits（tensor, shape=[N, num_classes]），可选
    
    返回：
        ba: 平衡准确率（支持多分类）
        cm: 混淆矩阵
        tpr: 二分类的真正例率（多分类返回None）
        tnr: 二分类的真负例率（多分类返回None）
        auc: AUC 值（若提供 y_prob，否则为 None）
    """
    # 计算混淆矩阵
    num_classes = int(y_true.max() + 1)
    cm = torchmetrics.functional.confusion_matrix(
        preds=y_pred,
        target=y_true,
        task='multiclass',
        num_classes=num_classes
    )

    # 平衡准确率（balanced accuracy）
    recalls = cm.diag() / (cm.sum(dim=1) + 1e-6)
    ba = recalls.nanmean()

    # 二分类的 TPR/TNR
    tpr = tnr = None
    if num_classes == 2:
        tn, fp, fn, tp = cm.flatten()
        tpr = tp / (tp + fn + 1e-6)
        tnr = tn / (tn + fp + 1e-6)

    # AUC 计算（若提供 y_prob）
    auc = None
    if y_prob is not None:
        if num_classes == 2:
            # 二分类：取阳性类别的概率
            pos_prob = y_prob[:, 1]
            auc = torchmetrics.functional.auroc(
                preds=pos_prob,
                target=y_true,
                task='binary'
            )
        else:
            # 多分类
            auc = torchmetrics.functional.auroc(
                preds=y_prob,
                target=y_true,
                task='multiclass',
                num_classes=num_classes,
                average='macro'  # 可选：macro/micro/weighted
            )

    return ba, cm, tpr, tnr, auc

def cal_F1_score(y_true, y_pred):
    """
    计算加权F1分数（支持多分类）
    输入应为类别标签（shape: [n_samples]）
    """
    f1 = torchmetrics.F1Score(
        task='multiclass',
        num_classes=int(y_true.max() + 1),
        average='weighted'
    ).to(y_true.device)
    return f1(y_pred, y_true)
