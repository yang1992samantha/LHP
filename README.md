## LHP: Logical Hypergraph Link Prediction  

### Introduction

 This is the PyTorch implementation of the LHP model.

### Run

For example,this command train and test a LHP model on CMHR dataset

python main.py --data "medical" --dataset "splits/1" --epoch 5000 --task "multi_classifier" --embedding "word2vector" --embed_dim_in 768

 Check argparse configuration at config.py for more arguments and more details. 

### Dataset

There are four datasets used in this paper,CMHR,iAF1260b,iJO1366,USPTO

In this paper, a logical hypergraph is proposed to express the directed high-order relationships in the medical domain, and a CMHR dataset is constructed.----"./medical",including two initialization embeddings(Bert,Word2vector),entity list, train data,valid data,test data.

iAF1260b  dataset          M. Zhang, Z. Cui, S. Jiang, and Y. Chen, "Beyond link prediction: Predicting hyperlinks in adjacency space" 

iJO1366  dataset          M. Zhang, Z. Cui, S. Jiang, and Y. Chen, "Beyond link prediction: Predicting hyperlinks in adjacency space" 

USPTO  dataset         Jin, W., Coley, C., Barzilay, R., Jaakkola, T., " Predicting Organic Reac-tion Outcomes with Weisfeiler-Lehman Network"

