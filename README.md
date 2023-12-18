# SynerGNet
SynerGNet is a machine learning tool to predict synergy effects of anti-cancer drug combinations. Given the gene expression, copy number variation (CNV), mutaion types, and gene ontology (GO) terms of the cell lines and drug-protein association scores of the paired drugs, the model is able to predict whether the paried drugs are synergistic or antagnistic against the cancer cell line.
## Dependencies
1. pytorch 1.10.0
2. torch_geometric 2.0.2
3. numpy 1.19.2
4. sklearn 0.23.2
5. pandas 1.1.3
6. CUDA 11.1
## Data preparation
The input to SynerGNet is graph representation of synergy instances. Each input graph is created through the following procedure:

1. Create full-size graph by mapping gene expression, CNV, mutation, GO terms and drug-protein association score onto PPI network. Each graph is represented using a node table and an edge table.
2. Knowledge-based graph reduction to increase the topology diversity and reduce computaion burden.

The source code of graph reduction and format of node tables and edge tables are provided in the repository https://github.com/MengLiu90/Two_set_graph_reduction_for_SynerGNet.

The reduced graphs (represented as node tables and edge tables) are further converted into a hierarchical format for further processing in the SynerGNet. ```./Dataset/Input_data/22RV1_CIDs15951529_CIDs54751698.h5``` provides an example of the input graph to SynerGNet.
```./Dataset/h5py_data.py``` provides the code to create the .h5 file from node table and edges table.

## 5-fold cross-validation of SynerGNet
The ```./5-fold cross-validation of SynerGNet/``` directory holds both the trained model and prediction results for SynerGNet in the context of a 5-fold cross-validation.

Execute the script ```./5-fold cross-validation of SynerGNet/SynerGNet_performance.py``` to replicate the SynerGNet performance table presented in the paper.
## Prediction using the trained model
Execute the script ```SynerGNet_Prediction.py``` to make predictions of drug synergy using the trained SynerGNet model.

Directly run ```python SynerGNet_Prediction.py``` will produce the prediction results of synergy instances from DrugCombDB, i.e., the validation dataset used in the paper.

To make prediction of your own synergy data, put the .h5 format graphs created from your synergy dataset in ```/Dataset/DrugCombDB/h5py_synergy_data/``` directory, then excute the python script. 

Note: two types of trained SynerGNet are provided in ```./Trained_models/``` directory: model trained on augmented data ```./Trained_models/SynerGNet_trained_on_aug.pth``` and model trained on original data ```./Trained_models/SynerGNet_trained_on_org.pth```. Which model to be used for prediction can be simply chosen within the ```SynerGNet_Prediction.py``` code by unquoting the selected model. One can try and see the difference of prediction performance of these two models. 
## Train your own model
