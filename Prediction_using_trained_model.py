import os, sys
import h5py
import torch
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GENConv, GraphConv, JumpingKnowledge, GATv2Conv, GINConv
# from torch.nn import Sequential, ReLU, Linear, Embedding
# from torch_geometric.nn import JumpingKnowledge, Set2Set, BatchNorm, GraphNorm, LayerNorm
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import time
import random
import shutil
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

test_dir = 'Dataset/DrugCombDB/h5py_synergy_data'
testdata_dir_p = 'Dataset/DrugCombDB/processed_validation_data/'
if not os.path.isdir(testdata_dir_p):
    os.mkdir(testdata_dir_p)
df = pd.read_csv('Dataset/DrugCombDB/drugcombs_synergy_data.csv')
class DatasetTest(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DatasetTest, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['TestData.pt']

    def _download(self):
        pass

    def process(self):
        data_list = []
        for file in tqdm(os.listdir(test_dir)):
            with h5py.File(os.path.join(test_dir, file), 'r') as f:
                X = f['X'][()]
                eI = f['eI'][()]
                eAttr = f['edge_weight'][()]
                y = f['y'][()]
            X = torch.tensor(X, dtype=torch.float32)
            eI = torch.tensor(eI, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            eAttr = torch.tensor(eAttr, dtype=torch.float32)
            instance = file
            data = Data(x=X, edge_index=eI, edge_attr=eAttr, y=y, ins=instance)
            data_list.append(data)
        data, slice = self.collate(data_list)
        torch.save((data, slice), self.processed_paths[0])


dataset_test = DatasetTest(root=testdata_dir_p)

# dataset_test = dataset_test.shuffle()
print('validation instances', len(dataset_test))

num_node_features = 218
num_classes = 2
class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()

        self.num_layers = num_layers

        # Create a list to hold SAGEConv layers
        self.conv_layers = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(GENConv(in_channels=num_node_features, out_channels=512,
                                                aggr='power', p=1, learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))
            else:
                self.conv_layers.append(GENConv(in_channels=512, out_channels=512, aggr='power', p=1,
                                                learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))

        self.batch_norm_layers = torch.nn.ModuleList([BatchNorm1d(512) for _ in range(num_layers)])

        # Create JumpingKnowledge module
        self.jk = JumpingKnowledge(mode='max')

        self.fc1 = Linear(2 * 512, 512)
        self.batch_norm4 = BatchNorm1d(512)
        self.dropout = Dropout(0.65)
        self.fc2 = Linear(512, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        node_representations = []

        # Forward pass through GATv2Conv layers
        for i in range(self.num_layers):
            # print('layer {}'.format(i))
            # print('shape of x before conv', x.shape, batch.shape)
            x = self.conv_layers[i](x, edge_index)
            # print('shape of x after conv', x.shape, batch.shape)
            x = self.batch_norm_layers[i](x)
            x = F.relu(x)

            # Collect intermediate node representations
            node_representations.append(x)

        # Apply JumpingKnowledge aggregation to node representations
        # print('shape of x before jk', x.shape)
        x = self.jk(node_representations)

        # Concatenate global max-pooling and global average-pooling representations
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        # Continue with the rest of the layers as before
        x = self.fc1(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.dropout(x)
        # fc2_features = x
        logits = self.fc2(x)

        return logits

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (Linear, SAGEConv, BatchNorm1d)):
                module.reset_parameters()

model = Net(num_layers=2)
print('model:', model)
batch_size = 16
test_acc_all = 0
test_loss_all = 0
auc_all = 0
bac_all = 0
cm_all = np.zeros((2,2))
precision_all = 0
recall_all = 0
f1_all = 0
fpr_all = 0
mcc_all = 0
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
def CLassWeights():
    labels = df['class'].tolist()
    lal = torch.as_tensor(labels)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                    classes=np.unique(lal),
                                                                    y=lal.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights
criterion = torch.nn.CrossEntropyLoss(weight=CLassWeights(), reduction='mean')

df_record = pd.DataFrame(columns=['instances', 'label', 'prediction', 'probability'])

# Load model trained against augmented data
# saved_state_dict = torch.load('Trained_models/SynerGNet_trained_on_aug.pth')
# Load model trained against original data
saved_state_dict = torch.load('Trained_models/SynerGNet_trained_on_org.pth')
# Load the state_dict into the model
model.load_state_dict(saved_state_dict)

model.eval()
v_losses = 0
v_acc = 0
num_batch_test = 0
predictions = []
labels = []
instances_list = []
y_scores = []
y_probs_list = []
TPRS = []
base_fpr = np.linspace(0, 1, 101)
with torch.no_grad():
    for j, data in enumerate(test_loader):
        num_batch_test += 1
        output = model(data)
        t_loss = criterion(output, data.y.squeeze())
        # v_losses += data.num_graphs * t_loss.item()
        inter_loss = t_loss.data.cpu().numpy()
        v_losses += data.num_graphs * inter_loss

        y_pred = output.data.cpu().numpy().argmax(axis=1)
        y_probs = F.softmax(output.data, dim=1).cpu().numpy()
        y_score = y_probs[:, 1]
        y_true = data.y.squeeze().cpu().numpy()
        v_acc += accuracy_score(y_true, y_pred)
        ins_list = data.ins

        predictions += y_pred.tolist()
        labels += y_true.tolist()
        y_scores += y_score.tolist()
        y_probs_list += y_probs.tolist()
        instances_list += ins_list

test_losses = v_losses / len(df)
test_acc = v_acc / num_batch_test
conf_matx = confusion_matrix(labels, predictions)
matrix_result = conf_matx
fpr, tpr, thresholds = metrics.roc_curve(np.asarray(labels), np.asarray(y_scores), pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
AUC = roc_auc

bal_acc = balanced_accuracy_score(labels, predictions)
BAC = bal_acc

tpr = np.interp(base_fpr, fpr, tpr)
tpr[0] = 0.0
TPRS.append(tpr)

test_acc_all += test_acc
test_loss_all += test_losses
auc_all += AUC
bac_all += BAC
cm_all += matrix_result

TP = matrix_result[1][1]
TN = matrix_result[0][0]
FP = matrix_result[0][1]
FN = matrix_result[1][0]
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

print('test accuracy', test_acc)
print('test loss', test_losses)
print('confusion matrix', matrix_result)
print('auc', AUC)
print('bac', BAC)
print('precision', Precision)
print('recall', Recall)
print('f1', F1)
print('fpr', FPR)
print('mcc', mcc)





