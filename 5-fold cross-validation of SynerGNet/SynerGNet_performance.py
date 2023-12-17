import numpy as np
import os

epochs = 200
num_fold = 5
result_dirs = ['Trained on augmented data/Results record', 'Trained on original data/Results record']
i = 0
for result_dir in result_dirs:
    cm_all = np.zeros((epochs, 2, 2))
    bac_all = np.zeros(epochs)
    auc_all = np.zeros(epochs)
    acc_all = np.zeros(epochs)
    avg_bac = 0

    acc_b = 0
    trainacc_b = 0
    auc_b = 0
    bac_b = 0
    train_bac_b = 0
    loss_b_all = 0
    cm_b = np.zeros((2, 2))
    precision_all = 0
    recall_all = 0
    f1_all = 0
    fpr_all = 0
    mcc_all = 0
    tpr_b = []

    for fd in range(num_fold):
        cm = np.load(result_dir + f'/confusionMatrix_{fd}.npy')
        bac = np.load(result_dir + f'/balanced_accuracy_{fd}.npy')
        train_bac = np.load(result_dir + f'/balanced_accuracy_train_{fd}.npy')
        auc = np.load(result_dir + f'/AUC_{fd}.npy')
        tpr = np.load(result_dir + f'/tpr_{fd}.npy')
        cm_all += cm
        bac_all += bac
        auc_all += auc
        # performance based on bac
        best_ep_bac = np.argmax(bac)
        tpr_b.append(tpr[best_ep_bac])
        auc_b += auc[best_ep_bac]

        bac_b += bac[best_ep_bac]
        train_bac_b += train_bac[best_ep_bac]
        cm_b += cm[best_ep_bac]

        TP = cm[best_ep_bac][1][1]
        TN = cm[best_ep_bac][0][0]
        FP = cm[best_ep_bac][0][1]
        FN = cm[best_ep_bac][1][0]

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        FPR = FP / (FP + TN)
        mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        precision_all += Precision
        recall_all += Recall
        f1_all += F1
        fpr_all += FPR
        mcc_all += mcc

    if i == 0:
        print('=============== performance of SynerGNet trained on augmented data ===================')
    else:
        print('=============== performance of SynerGNet trained on original data ===================')
    print('bac', bac_b / num_fold)
    print('auc', auc_b / num_fold)
    print('train bac', train_bac_b / num_fold)
    print('difference b/t train bac and test bac', train_bac_b / num_fold - bac_b / num_fold)
    print('confusion matrix', cm_b)
    print('precision', precision_all / num_fold)
    print('recall', recall_all / num_fold)
    print('f1', f1_all / num_fold)
    print('fpr', fpr_all / num_fold)
    print('mcc', mcc_all / num_fold)
    np.save(os.path.join(result_dir, 'tpr_best_bac.npy'), tpr_b)
    i += 1


