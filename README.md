# GTCM:Identifying cancer driver genes using a neural network framework with cross-attention mechanism  
GTCM is a new framework combining Transformer with cross-attention. The input data of GTCM include PPI network, pathway co-occurrence association network (PCAN), gene functional similarity network (GFSN) and gene feature matrix X. Firstly, GTCM learns gene feature representations from the input data by using a three-layer GraphSAGE network. Then, GTCM adopts a Transformer encoding layer with cross-attention to enhance the feature representations learned from the GraphSAGE network. Finally, GTCM uses MLP classifier to predict cancer driver genes and outputs the probability of each gene being cancer driver gene. The overview of GTCM is shown as follows.  
![E_SAGE_N](https://github.com/user-attachments/assets/ccaba49b-05ad-4131-8094-cadce08bb938)
## Requirements  
The project is written in Python 3.10, and all experiments are performed on an Ubuntu server equipped with an Intel Xeon CPU (2.4GHz, 128G RAM) and an Nvidia RTX 4090D GPU (24G GPU RAM). To speed up the training process, training must be performed on a GPU, but a standard computer without a GPU can also be used (at the expense of more training time). All GTCM implementations are based on PyTorch and PyTorch Geometric. GTCM requires the following dependencies:  
  * python == 3.10
  * torch = 2.0.0
  * torch-geometric = 2.1.2
## Reproducibility  
The results can be reproduced using the following `GTCM_5fold.py` script, and the following `GTCM_train_predict.py` script is used to train the GTCM model n times and generate a ranked list of predicted driver genes.  
## Preprocessing your own data  
See [preprocess_data](#preprocess_data) to learn how to process your own data and prepare it for training GTCM.  
* In our study, we followed the data preprocessing steps described in [EMOGI](https://github.com/schulter/EMOGI), and the data preprocessing code was derived from EMOGI. We also referred to the preprocessing steps for gene relationship networks in [MODIG](https://github.com/zjupgx/modig.git) and [MNGCL](https://github.com/weiba/MNGCL.git), and the positive and negative sample division processing steps in [HGDC](https://github.com/NWPU-903PR/HGDC.git).
