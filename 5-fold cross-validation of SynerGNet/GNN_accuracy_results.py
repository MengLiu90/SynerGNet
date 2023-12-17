import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
df = pd.read_csv('C:/Users/mengm/PycharmProjects/pythonProject1/drugCombo/'
                 'Data_combo/synergy_classification_original_ts_combined.csv')
c = dict(Counter(df['class']))
print(c)

epochs = 200
# ep = 59
cm_all = np.zeros((epochs, 2, 2))
bac_all = np.zeros(epochs)
auc_all = np.zeros(epochs)
acc_all = np.zeros(epochs)
avg_bac = 0

acc_l = 0
auc_l = 0
bac_l = 0
cm_l = np.zeros((2,2))

acc_b = 0
trainacc_b = 0
auc_b = 0
bac_b = 0
train_bac_b = 0
loss_b_all = 0
cm_b = np.zeros((2,2))
precision_all = 0
recall_all = 0
f1_all = 0
fpr_all = 0
mcc_all = 0
tpr_b = []

acc_a = 0
auc_a = 0
bac_a = 0
cm_a = np.zeros((2,2))

acc_au = 0
auc_au = 0
bac_au = 0
cm_au = np.zeros((2,2))
result_dir = 'results_GEN_x32_sp2' # best GNN model trained with augmented data
# result_dir = 'results_rd_split_mc_GEN_tune_org_new' # GNN model trained with original data
num_fold = 5
for fd in range(num_fold):
    cm = np.load(result_dir + f'/confusionMatrix_{fd}.npy')
    bac = np.load(result_dir + f'/balanced_accuracy_{fd}.npy')
    train_bac = np.load(result_dir + f'/balanced_accuracy_train_{fd}.npy')
    acc = np.load(result_dir + f'/testAcc_{fd}.npy')
    acc_train = np.load(result_dir + f'/trainAcc_{fd}.npy')
    auc = np.load(result_dir + f'/AUC_{fd}.npy')
    loss = np.load(result_dir + f'/testLoss_{fd}.npy')
    tpr = np.load(result_dir + f'/tpr_{fd}.npy')
    cm_all += cm
    bac_all += bac
    auc_all += auc
    acc_all += acc

    # performance based on loss
    best_ep_loss = np.argmin(loss)
    least_loss = loss[best_ep_loss]
    print('at epoch {}, the smallest loss is {}'.format(best_ep_loss, least_loss))
    acc_l += acc[best_ep_loss]
    auc_l += auc[best_ep_loss]
    bac_l += bac[best_ep_loss]
    cm_l += cm[best_ep_loss]

    # performance based on bac
    best_ep_bac = np.argmax(bac)
    loss_b = loss[best_ep_bac]
    loss_b_all += loss_b
    print('based on bac at epoch {}, the loss is {}, the highest bac in this fold is {}'.format(best_ep_bac, loss_b,
                                                                                                bac[best_ep_bac]))
    tpr_b.append(tpr[best_ep_bac])
    acc_b += acc[best_ep_bac]
    trainacc_b += acc_train[best_ep_bac]
    auc_b += auc[best_ep_bac]
    print('#################################')
    print(auc[best_ep_bac])
    print('###################################')
    bac_b += bac[best_ep_bac]
    train_bac_b += train_bac[best_ep_bac]
    cm_b += cm[best_ep_bac]

    TP = cm[best_ep_bac][1][1]
    TN = cm[best_ep_bac][0][0]
    FP = cm[best_ep_bac][0][1]
    FN = cm[best_ep_bac][1][0]
    # print('TP', TP)
    # print('TN', TN)
    # print('FP', FP)
    # print('FN', FN)

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    FPR = FP / (FP + TN)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    precision_all +=Precision
    recall_all += Recall
    f1_all += F1
    fpr_all += FPR
    mcc_all += mcc

    # performance based on acc
    best_ep_acc = np.argmax(acc)
    loss_a = loss[best_ep_acc]
    print('based on acc at epoch {}, the loss is {}'.format(best_ep_acc, loss_a))

    auc_a += auc[best_ep_acc]
    bac_a += bac[best_ep_acc]
    cm_a += cm[best_ep_acc]
    print('fold {}'.format(fd))
    # print('accuracy', acc[best_ep_acc])

    cm_ep = cm[best_ep_acc]
    acc_cm = (cm_ep[0][0] + cm_ep[1][1]) / (cm_ep[0][0] + cm_ep[0][1] + cm_ep[1][0] + cm_ep[1][1])
    print('accuracy', acc_cm)
    print('auc', auc[best_ep_acc])
    print('bac', bac[best_ep_acc])
    print('confusion matrix', cm[best_ep_acc])
    acc_a += acc_cm

    # performance based on auc
    best_ep_auc = np.argmax(auc)
    loss_au = loss[best_ep_auc]
    # print('based on auc at epoch {}, the loss is {}'.format(best_ep_auc, loss_au))
    acc_au += acc[best_ep_auc]
    auc_au += auc[best_ep_auc]
    bac_au += bac[best_ep_auc]
    cm_au += cm[best_ep_auc]

    plt.figure()
    # plt.plot(range(len(tr_losses)), tr_losses, color='blue')
    plt.plot(range(len(loss)), loss, color='red')
    plt.legend(['Test loss'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(result_dir, 'test_loss_fold_{}.png'.format(fd)))
    plt.show()

    # cm_ep = cm[ep]
    # bac_cm = (cm_ep[0][0] / (cm_ep[0][0] + cm_ep[0][1]) + cm_ep[1][1] / (cm_ep[1][0] + cm_ep[1][1])) / 2
    # acc_cm = (cm_ep[0][0] + cm_ep[1][1]) / (cm_ep[0][0] + cm_ep[0][1] + cm_ep[1][0] + cm_ep[1][1])
    #
    # avg_bac += bac[ep]

    # print(cm[ep])
    # print('accuracy',acc[ep])
    # print('accuracy from confusion matrix',acc_cm)
    # print('auc', auc[ep])
    # print('bac', bac[ep])
    # print('bac from comfusion matrix', bac_cm)

print('=============== performance based on loss =================')
print('bac', bac_l/num_fold)
print('auc', auc_l/num_fold)
print('accuracy', acc_l/num_fold)
print('confusion matrix', cm_l)

print('=============== performance based on bac ===================')
print('bac', bac_b/num_fold)
print('auc', auc_b/num_fold)
print('accuracy', acc_b/num_fold)
print('train bac', train_bac_b/num_fold)
print('train accuracy', trainacc_b/num_fold)
print('difference b/t train bac and test bac', train_bac_b/num_fold - bac_b/num_fold)
print('difference b/t train acc and test acc', trainacc_b/num_fold - acc_b/num_fold)
print('confusion matrix', cm_b)
print('loss', loss_b_all/num_fold)
print('precision', precision_all/num_fold)
print('recall', recall_all/num_fold)
print('f1', f1_all/num_fold)
print('fpr', fpr_all/num_fold)
print('mcc', mcc_all/num_fold)
np.save(os.path.join(result_dir, 'tpr_best_bac.npy'), tpr_b)

print('=============== performance based on acc ====================')
print('bac', bac_a/num_fold)
print('auc', auc_a/num_fold)
print('accuracy', acc_a/num_fold)
print('confusion matrix', cm_a)

print('=============== performance based on auc ====================')
print('bac', bac_au/num_fold)
print('auc', auc_au/num_fold)
print('accuracy', acc_au/num_fold)
print('confusion matrix', cm_au)
print('==============================================================')

average_bac_all = bac_all/num_fold
average_auc_all = auc_all/num_fold
average_acc_all = acc_all/num_fold
largest_bac_idx = np.argmax(average_bac_all)
largest_bac = average_bac_all[largest_bac_idx]
print(largest_bac, largest_bac_idx)
top = 5
ind = np.argpartition(average_bac_all, -top)[-top:]
top_bac = average_bac_all[ind]
print(ind)
print('top bac',top_bac)
print('auc',average_auc_all[ind])
print('accuracy',average_acc_all[ind])

cfm0 = np.load(result_dir + '/confusionMatrix_0.npy')
# cfm0_s = cfm0[ind]
cfm1 = np.load(result_dir + '/confusionMatrix_1.npy')
# cfm1_s = cfm1[ind]
cfm2 = np.load(result_dir + '/confusionMatrix_2.npy')
# cfm2_s = cfm2[ind]
cfm3 = np.load(result_dir + '/confusionMatrix_3.npy')
# cfm3_s = cfm3[ind]
cfm4 = np.load(result_dir + '/confusionMatrix_4.npy')
# cfm4_s = cfm4[ind]
for indx in ind:
    print('epoch', indx)
    # print('cm of fold 0','\n',cfm0[indx])
    # print('cm of fold 1','\n',cfm1[indx])
    # print('cm of fold 2','\n',cfm2[indx])
    # print('cm of fold 3','\n',cfm3[indx])
    # print('cm of fold 4','\n',cfm4[indx])
    print('cm in this epoch', '\n', cfm0[indx]+cfm1[indx]+cfm2[indx]+cfm3[indx]+cfm4[indx])

# selected_auc = average_auc_all[ep]
# selected_accuracy = average_acc_all[ep]
# selected_bac = average_bac_all[ep]
# selected_cm = cm_all[ep]
# print('best auc in all 300 epochs', selected_auc)
# print('best accuracy in all 300 epochs', selected_accuracy)
# print('best bac in all 300 epochs', selected_bac)
# print('calculated bac', avg_bac/5)
# print('best comfusion matrix', selected_cm)