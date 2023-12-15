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
The input to SynerGNet is graph representation of synergy instances. Each input graph is created throught the following procedure:

1. Create full-size graph by mapping gene expression, CNV, mutation, GO terms and drug-protein association score onto PPI network.
2. Knowledge-based graph reduction to increase the topology diversity and reduce computaion burden.
  
## Prediction using the trained model

## Train your own model
