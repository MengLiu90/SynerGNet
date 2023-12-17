import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os

gnn_aug_dir = 'results_GEN_x32_sp2'
gnn_org_dir = 'results_rd_split_mc_GEN_tune_org_new'
rf_aug_dir = 'RF_Results_aug_3_used'
rf_org_dir = 'RF_Results_org_2_used'
figure_dir = 'Figures'
gnn_aug_tpr_all = np.load(gnn_aug_dir + '/tpr_best_bac.npy')
gnn_org_tpr_all = np.load(gnn_org_dir + '/tpr_best_bac.npy')

base_fpr = np.linspace(0, 1, 101)
folds = 5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=300)
##### RF ROC plot #################
rf_org_tpr_all = []
rf_aug_tpr_all = []
for fd in range(folds):
    rf_org_fpr = np.load(rf_org_dir + '/test_fpr_fold_{}.npy'.format(fd))
    rf_org_tpr = np.load(rf_org_dir + '/test_tpr_fold_{}.npy'.format(fd))
    rf_org_tpr_interpolated = np.interp(base_fpr, rf_org_fpr, rf_org_tpr)
    rf_org_tpr_interpolated[0] = 0.0
    ax1.plot(base_fpr, rf_org_tpr_interpolated, lw=1, alpha=0.5, color='m')
    rf_org_tpr_all.append(rf_org_tpr_interpolated)

    rf_aug_fpr = np.load(rf_aug_dir + '/test_fpr_fold_{}.npy'.format(fd))
    rf_aug_tpr = np.load(rf_aug_dir + '/test_tpr_fold_{}.npy'.format(fd))
    rf_aug_tpr_interpolated = np.interp(base_fpr, rf_aug_fpr, rf_aug_tpr)
    rf_aug_tpr_interpolated[0] = 0.0
    rf_aug_tpr_all.append(rf_aug_tpr_interpolated)
    ax1.plot(base_fpr, rf_aug_tpr_interpolated, lw=1, alpha=0.5, color='b')


rf_org_tprs_all = np.asarray(rf_org_tpr_all)
rf_org_tpr_all_mean = rf_org_tprs_all.mean(axis=0)
rf_org_auc_all_mean = metrics.auc(base_fpr, rf_org_tpr_all_mean)
print('RF original Mean ROC (AUC = %0.4f)' % (rf_org_auc_all_mean))
ax1.plot(base_fpr, rf_org_tpr_all_mean, color='m',
         label='Original data',
         lw=3, alpha=1)

rf_aug_tprs_all = np.asarray(rf_aug_tpr_all)
rf_aug_tpr_all_mean = rf_aug_tprs_all.mean(axis=0)
rf_aug_auc_all_mean = metrics.auc(base_fpr, rf_aug_tpr_all_mean)
print('RF augmented Mean ROC (AUC = %0.4f)' % (rf_aug_auc_all_mean))
ax1.plot(base_fpr, rf_aug_tpr_all_mean, color='b',
         label='Augmented data',
         lw=3, alpha=1)
ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
ax1.set_title('A', fontsize=15)
# ax1.set(xlabel='FPR', ylabel='TPR', fontsize=20)
ax1.set_xlabel('FPR', fontsize=15) # X label
ax1.set_ylabel('TPR', fontsize=15) # Y label
ax1.set_aspect('equal', adjustable='box')
ax1.tick_params(axis='both', which='major', labelsize=14)

for fd in range(folds):
    tpr = gnn_aug_tpr_all[fd]
    auc = metrics.auc(base_fpr, tpr)
    print('auc', auc)
    plt.plot(base_fpr, tpr, lw=1, alpha=0.5, color='b')
    tpr_o = gnn_org_tpr_all[fd]
    auc_org = metrics.auc(base_fpr, tpr_o)
    plt.plot(base_fpr, tpr_o, lw=1, alpha=0.5, color='m')
tprs = np.asarray(gnn_aug_tpr_all)
mean_tprs = tprs.mean(axis=0)
mean_auc = metrics.auc(base_fpr, mean_tprs)
# plt.plot(base_fpr, mean_tprs, color='b',
#          label='x32 augmented data + original data',
#          lw=3, alpha=1)
ax2.plot(base_fpr, mean_tprs, color='b',
         label='Augmented data',
         lw=3, alpha=1)
tprs_o = np.asarray(gnn_org_tpr_all)
mean_tprs_o = tprs_o.mean(axis=0)
mean_auc_o = metrics.auc(base_fpr, mean_tprs_o)
ax2.plot(base_fpr, mean_tprs_o, color='m',
         label='Original data',
         lw=3, alpha=1)
ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
ax2.set_title('B', fontsize=15)
# ax1.set(xlabel='FPR', ylabel='TPR', fontsize=20)
ax2.set_xlabel('FPR', fontsize=15) # X label
ax2.set_ylabel('TPR', fontsize=15) # Y label
ax2.set_aspect('equal', adjustable='box')
ax2.tick_params(axis='both', which='major', labelsize=14)

print('GNN aug mean auc', mean_auc)
print('GNN org mean auc', mean_auc_o)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1]
# Check the length of handles and labels to make sure your order is valid
if len(handles) >= 2 and len(labels) >= 2:  # Change 2 to the number of items you expect
    # Reorder and create the legend
    ax1.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', prop={'size': 14})
    ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower right', prop={'size': 14})
else:
    # If there are not enough legend items, just create the legend without reordering
    ax1.legend(loc='lower right', prop={'size': 14})
    ax2.legend(loc='lower right', prop={'size': 14})



# Show the plot
plt.savefig(os.path.join(figure_dir, 'ROC_Plot_complete.png'))
plt.show()







