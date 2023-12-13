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
# from Focal_loss import *
# from radam import RAdam
df = pd.read_csv('/work/lmengm1/original_synergy_data/synergy_classification_original_ts_combined.csv')
# df_aug = pd.read_csv('/work/lmengm1/Test_aug/sampled_aug_data_64_remedy.csv')

# seed = random.randint(1, 500)
seed = 42
print('random seed', seed)
# Set seed for PyTorch (for both CPU and CUDA)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Set seeds for random and numpy
random.seed(seed)
np.random.seed(seed)

# sampled_df = pd.DataFrame(columns=df_aug.columns)
# groups = df_aug.groupby(['track'])
# for _, gp in groups:
#     sampled_ins = gp.sample(n=8, random_state=seed, replace=True)
#     sampled_df = sampled_df.append(sampled_ins)
# sampled_df.to_csv('/work/lmengm1/Test_aug/aug_samples_for_train/sampled_aug_data_for_each_org_ins_remedy_x8.csv', index=False)
# df_sample = pd.read_csv('/work/lmengm1/Test_aug/aug_samples_for_train/sampled_aug_data_for_each_org_ins_remedy.csv')
# df_sample['instances'] = df_sample['Cell_Line'] + '_' + df_sample['CID_1'] + '_' + df_sample['CID_2']

# h5py data source directory
org_h5py_source = '/work/lmengm1/Test_GO_embeddings/h5py_synergy_ALL_Features_column_reordered'
# aug_h5py_source = '/work/lmengm1/Test_aug/h5py_data_sampled_remedy_64'

# directories
save_dir = '/work/lmengm1/Model_Selection/Final_model/results_rd_split_mc_GEN_tune_org'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

test_dir = '/work/lmengm1/Model_Selection/Final_model/Data_test_mc_GEN_tune_org'
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

train_dir = '/work/lmengm1/Model_Selection/Final_model/Data_train_mc_GEN_tune_org'
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

testdata_dir_p = '/work/lmengm1/Model_Selection/Final_model/DataForModelTest_mc_GEN_tune_org'
if not os.path.isdir(testdata_dir_p):
    os.mkdir(testdata_dir_p)

traindata_dir_p = '/work/lmengm1/Model_Selection/Final_model/DataForModelTrain_mc_GEN_tune_org'
if not os.path.isdir(traindata_dir_p):
    os.mkdir(traindata_dir_p)

class Net(torch.nn.Module):
    def __init__(self, num_layers):
        super(Net, self).__init__()

        self.num_layers = num_layers

        # Create a list to hold SAGEConv layers
        self.conv_layers = torch.nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(GENConv(in_channels=dataset_test.num_node_features, out_channels=512,
                                                aggr='power', p=1, learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))
            else:
                self.conv_layers.append(GENConv(in_channels=512, out_channels=512, aggr='power', p=1,
                                                learn_p=True, msg_norm=True, learn_msg_scale=True,
                                                norm='layer', num_layers=1))

        self.batch_norm_layers = torch.nn.ModuleList([BatchNorm1d(512) for _ in range(num_layers)])

        self.fc1 = Linear(2 * 512, 512)
        self.batch_norm4 = BatchNorm1d(512)
        self.dropout = Dropout(0.65)
        self.fc2 = Linear(512, dataset_test.num_classes)

        # Create JumpingKnowledge module
        self.jk = JumpingKnowledge(mode='max')

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
        logits = self.fc2(x)

        return logits

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (Linear, SAGEConv, BatchNorm1d)):
                module.reset_parameters()


def CLassWeights():
    labels = df['class'].tolist()
    lal = torch.as_tensor(labels)
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                    classes=np.unique(lal),
                                                                    y=lal.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights

# def SampledInstances(org_track): #train instances from aug data
#
#     df_spl_train = df_sample.loc[df_sample['track'].isin(org_track)]
#     train_ins = df_spl_train['instances'].tolist()
#     for ins in train_ins:
#         string = ins.replace('\r', '')
#         cell_drug_list = string.split('_')
#         cell = cell_drug_list[0]
#         drug1 = cell_drug_list[1]
#         drug2 = cell_drug_list[2]
#         shutil.copy(os.path.join(aug_h5py_source, f'{cell}_{drug1}_{drug2}.h5'),
#                     train_dir)

def TrainInstances(org_track): #train instances from original data
    for ins in org_track:
        string = ins.replace('\r', '')
        cell_drug_list = string.split('_')
        cell = cell_drug_list[0]
        drug1 = cell_drug_list[1]
        drug2 = cell_drug_list[2]
        shutil.copy(os.path.join(org_h5py_source, f'{cell}_{drug1}_{drug2}.h5'),
                    train_dir)

def TestInstances(test_track):
    for ins in test_track:
        string = ins.replace('\r', '')
        cell_drug_list = string.split('_')
        cell = cell_drug_list[0]
        drug1 = cell_drug_list[1]
        drug2 = cell_drug_list[2]
        shutil.copy(os.path.join(org_h5py_source, f'{cell}_{drug1}_{drug2}.h5'),
                    test_dir)

num_fold = 5
epochs = 200
skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)
fold_cnt = 0

for train_index, test_index in skf.split(df, df.iloc[:, 5]):
    train_set = df.iloc[train_index, :]
    test_set = df.iloc[test_index, :]
    org_track = train_set.instances.tolist()
    test_track = test_set.instances.tolist()

    ## clear all the previous data
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))

    for f in os.listdir(train_dir):
        os.remove(os.path.join(train_dir, f))

    testdata_dir = os.path.join(testdata_dir_p, 'processed')
    if os.path.exists(testdata_dir):
        shutil.rmtree(testdata_dir)
    else:
        print("no processed data in DataForModelTest folder")
    traindata_dir = os.path.join(traindata_dir_p, 'processed')
    if os.path.exists(traindata_dir):
        shutil.rmtree(traindata_dir)
    else:
        print("no processed data in DataForModelTrain folder")

    print('all previous data removed')
    start = time.time()
    # SampledInstances(org_track)
    tim1 = time.time()
    # print('take {} for sampled augmented instances ready'.format(tim1 - start))
    TrainInstances(org_track)
    tim2 = time.time()
    print('take {} for original train instances ready'.format(tim2 - tim1))
    TestInstances(test_track)
    tim3 = time.time()
    print('take {} for test instances ready'.format(tim3 - tim1))

    class DatasetTrain(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None):
            super(DatasetTrain, self).__init__(root, transform, pre_transform)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return []

        @property
        def processed_file_names(self):
            return ['TrainingData.pt']

        def _download(self):
            pass

        def process(self):
            data_list = []
            for file in tqdm(os.listdir(train_dir)):
                with h5py.File(os.path.join(train_dir, file), 'r') as f:
                    X = f['X'][()]
                    eI = f['eI'][()]
                    eAttr = f['edge_weight'][()]
                    y = f['y'][()]
                X = torch.tensor(X, dtype=torch.float32)
                eI = torch.tensor(eI, dtype=torch.long)
                y = torch.tensor(y, dtype=torch.long)
                eAttr = torch.tensor(eAttr, dtype=torch.float32)
                data = Data(x=X, edge_index=eI, edge_attr=eAttr, y=y)
                data_list.append(data)
            data, slice = self.collate(data_list)
            torch.save((data, slice), self.processed_paths[0])

    dataset_train = DatasetTrain(root=traindata_dir_p)
    dataset_train = dataset_train.shuffle()
    print('training instances', len(dataset_train))

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
                data = Data(x=X, edge_index=eI, edge_attr=eAttr, y=y)
                data_list.append(data)
            data, slice = self.collate(data_list)
            torch.save((data, slice), self.processed_paths[0])


    dataset_test = DatasetTest(root=testdata_dir_p)
    dataset_test = dataset_test.shuffle()
    print('testing instances', len(dataset_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_layers=2)
    model.reset_parameters()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.000001)
    criterion = torch.nn.CrossEntropyLoss(weight=CLassWeights(), reduction='mean')
    # criterion = FocalLoss(gamma=1)

    print('===================Fold {} starts==================='.format(fold_cnt))
    best_balanced_accuracy = 0.0
    best_model_state = None

    batch_size = 16
    if ((len(dataset_train) % batch_size) == 1) | ((len(dataset_test) % batch_size) == 1):
        batch_size = 16+1
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    tr_losses = np.zeros(epochs)
    tr_acc = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    test_acc = np.zeros(epochs)
    matrix_result = np.zeros((epochs, 2, 2))
    AUC = np.zeros(epochs)
    BAC = np.zeros(epochs)  # balanced accuracy
    BAC_train = np.zeros(epochs)
    TPRS = []
    base_fpr = np.linspace(0, 1, 101)

    # # model.apply(reset_weights)
    # for name, module in model.named_children():
    #     if name != 'dropout':
    #         print('resetting {}'.format(name))
    #         module.reset_parameters()
    # best_auc = 0.7
    for ep in range(epochs):
        print('current epoch {}'.format(ep))
        model.train()
        losses = 0
        acc = 0
        num_batch = 0
        train_predictions = []
        train_labels = []
        for i, data in enumerate(train_loader):
            num_batch += 1
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.squeeze())
            loss.backward()
            optimizer.step()

            # losses += data.num_graphs * loss.item()
            itr_loss = loss.data.cpu().numpy()
            losses += data.num_graphs * itr_loss

            y_pred = output.data.cpu().numpy().argmax(axis=1)
            y_probs = F.softmax(output.data, dim=1).cpu().numpy()
            y_true = data.y.squeeze().cpu().numpy()
            acc += accuracy_score(y_true, y_pred)

            train_predictions += y_pred.tolist()
            train_labels += y_true.tolist()

        tr_losses[ep] = losses / len(train_set)
        tr_acc[ep] = acc / num_batch
        bal_acc_train = balanced_accuracy_score(train_labels, train_predictions)
        BAC_train[ep] = bal_acc_train

        model.eval()
        v_losses = 0
        v_acc = 0
        num_batch_test = 0
        predictions = []
        labels = []
        y_scores = []
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                num_batch_test += 1
                data = data.to(device)
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

                predictions += y_pred.tolist()
                labels += y_true.tolist()
                y_scores += y_score.tolist()

        test_losses[ep] = v_losses / len(test_set)
        test_acc[ep] = v_acc / num_batch_test
        conf_matx = confusion_matrix(labels, predictions)
        matrix_result[ep] = conf_matx
        fpr, tpr, thresholds = metrics.roc_curve(np.asarray(labels), np.asarray(y_scores), pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        AUC[ep] = roc_auc

        bal_acc = balanced_accuracy_score(labels, predictions)
        BAC[ep] = bal_acc

        # select model with best bal_acc in current epoch
        if bal_acc > best_balanced_accuracy:
            best_balanced_accuracy = bal_acc
            best_model_state = model.state_dict()

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        TPRS.append(tpr)

    np.save(os.path.join(save_dir, 'trainAcc_{}.npy'.format(fold_cnt)), tr_acc)
    np.save(os.path.join(save_dir, 'testAcc_{}.npy'.format(fold_cnt)), test_acc)
    np.save(os.path.join(save_dir, 'trainLoss_{}.npy'.format(fold_cnt)), tr_losses)
    np.save(os.path.join(save_dir, 'testLoss_{}.npy'.format(fold_cnt)), test_losses)
    np.save(os.path.join(save_dir, 'confusionMatrix_{}.npy'.format(fold_cnt)), matrix_result)
    np.save(os.path.join(save_dir, 'AUC_{}.npy'.format(fold_cnt)), AUC)
    np.save(os.path.join(save_dir, 'tpr_{}.npy'.format(fold_cnt)), TPRS)
    np.save(os.path.join(save_dir, 'balanced_accuracy_{}.npy'.format(fold_cnt)), BAC)
    np.save(os.path.join(save_dir, 'balanced_accuracy_train_{}.npy'.format(fold_cnt)), BAC_train)

    # save the model with the best bac for each fold
    model_path = os.path.join(save_dir, 'best_model_fold_{}.pth'.format(fold_cnt))
    torch.save(best_model_state, model_path)

    plt.figure()
    plt.plot(range(len(tr_losses)), tr_losses, color='blue')
    plt.plot(range(len(test_losses)), test_losses, color='red')
    plt.legend(['Train loss', 'Test loss'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'Loss_fold_{}.png'.format(fold_cnt)))
    plt.show()

    plt.figure()
    plt.plot(range(len(tr_acc)), tr_acc, color='blue')
    plt.plot(range(len(test_acc)), test_acc, color='red')
    plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(save_dir, 'Acc_fold_{}.png'.format(fold_cnt)))
    plt.show()

    plt.figure()
    plt.plot(range(len(AUC)), AUC, color='blue')
    plt.plot(range(len(BAC)), BAC, color='green')
    plt.legend(['AUC values', 'balanced accuracy'], loc='lower right')
    plt.xlabel('Epoch')
    plt.savefig(os.path.join(save_dir, 'AUC_BAC_Plot_fold_{}.png'.format(fold_cnt)))
    plt.show()

    fold_cnt += 1





