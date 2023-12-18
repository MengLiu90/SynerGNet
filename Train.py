import os, sys
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold
import pandas as pd
import sklearn
import random
import shutil
from model import Net
from Create_torch_geometric_data import Dataset

def Train(synergy_file_path, h5py_dir_path):
    df = pd.read_csv(synergy_file_path)
    # df = pd.read_csv('./Dataset/DrugCombDB/drugcombs_synergy_data.csv')

    # seed = random.randint(1, 500)
    seed = 42
    # Set seed for PyTorch (for both CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set seeds for random and numpy
    random.seed(seed)
    np.random.seed(seed)

    # h5py data source directory
    # org_h5py_source = './Dataset/DrugCombDB/h5py_synergy_data'
    org_h5py_source = h5py_dir_path

    # directories
    save_dir = './results'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data_dir_p = './DataForModel'
    if not os.path.isdir(data_dir_p):
        os.mkdir(data_dir_p)

        ## clear the previous data
        testdata_dir = os.path.join(data_dir_p, 'processed')
        if os.path.exists(testdata_dir):
            shutil.rmtree(testdata_dir)
        else:
            print("no processed data in DataForModel folder")

    def CLassWeights():
        labels = df['class'].tolist()
        lal = torch.as_tensor(labels)
        class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                        classes=np.unique(lal),
                                                                        y=lal.numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        return class_weights

    epochs = 200

    dataset = Dataset(root=data_dir_p, org_h5py_source=org_h5py_source)
    dataset = dataset.shuffle()

    # Get the indices for the train and test split
    num_samples = len(dataset)
    indices = list(range(num_samples))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=seed)

    # Create train and test datasets
    train_set = dataset[train_indices]
    test_set = dataset[test_indices]

    # Optionally, shuffle the train and test datasets
    train_set.shuffle()
    test_set.shuffle()
    print(len(train_set), len(test_set))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_layers=2, num_node_features=dataset.num_node_features, num_classes=dataset.num_classes)
    model.reset_parameters()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.000001)
    criterion = torch.nn.CrossEntropyLoss(weight=CLassWeights(), reduction='mean')
    # criterion = FocalLoss(gamma=1)

    best_balanced_accuracy = 0.0
    best_model_state = None

    batch_size = 4
    if ((len(train_set) % batch_size) == 1) | ((len(test_set) % batch_size) == 1):
        batch_size = batch_size + 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
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

    np.save(os.path.join(save_dir, 'trainAcc.npy'), tr_acc)
    np.save(os.path.join(save_dir, 'testAcc.npy'), test_acc)
    np.save(os.path.join(save_dir, 'trainLoss.npy'), tr_losses)
    np.save(os.path.join(save_dir, 'testLoss_.npy'), test_losses)
    np.save(os.path.join(save_dir, 'confusionMatrix.npy'), matrix_result)
    np.save(os.path.join(save_dir, 'AUC.npy'), AUC)
    np.save(os.path.join(save_dir, 'tpr.npy'), TPRS)
    np.save(os.path.join(save_dir, 'balanced_accuracy.npy'), BAC)
    np.save(os.path.join(save_dir, 'balanced_accuracy_train.npy'), BAC_train)

    # save the model with the best bac for each fold
    model_path = os.path.join(save_dir, 'best_model_fold_{}.pth')
    torch.save(best_model_state, model_path)

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process data to train SynerGNet")

    # Add command-line arguments for CSV file path and H5PY directory path
    parser.add_argument("csv_file", help="Path to the synergy file")
    parser.add_argument("h5py_dir", help="Path to the H5PY directory")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the process_data function with the provided file paths
    Train(args.csv_file, args.h5py_dir)





