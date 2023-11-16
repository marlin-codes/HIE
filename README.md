## [Hyperbolic Representation Learning: Revisiting and Advancing (ICML 2023)](https://arxiv.org/abs/2306.09118)

**Authors**: Menglin Yang, Min Zhou, Rex Ying, Yankai Chen, Irwin King \
**Codes**: https://github.com/marlin-codes/HIE

## Abstract
The non-Euclidean geometry of hyperbolic spaces has recently garnered considerable attention in the realm of representation learning. Current endeavors in hyperbolic representation largely presuppose that the underlying hierarchies can be automatically inferred and preserved through the adaptive optimization process. This assumption, however, is questionable and requires further validation. In this work, we first introduce a position-tracking mechanism to scrutinize existing prevalent hyperbolic neural networks, revealing that the learned representations are sub-optimal and unsatisfactory. To address this, we propose a simple yet effective method, hyperbolic informed embedding (HIE), by incorporating cost-free hierarchical information deduced from the hyperbolic distance of the node to origin (i.e., induced hyperbolic norm) to advance existing hyperbolic neural networks. The proposed HIE method is both task-agnostic and model-agnostic, enabling its seamless integration with a broad spectrum of models and tasks. 

![Illustration of the basic idea of HIE method, which can be decomposed into two critical steps: root alignment and level-aware
stretching](idea.png)

## 1. Overview
This repository is an official PyTorch Implementation for "[Hyperbolic Representation Learning: Revisiting and Advancing (ICML 2023)](https://arxiv.org/abs/2306.09118)"



Note: this project is built upon [HGCN](https://github.com/HazyResearch/hgcn). We thank the authors for their wonderful work.

<a name="Environment"/>

## 2. Experimental Environment:

The code was developed and tested on the following python environment, other similar environment should also work.
```
python 3.8.10
scikit-learn 1.3.2
numpy 1.24.2
scipy 1.10.1
tqdm 4.61.2
torch                    2.0.0+cu118
torch-cluster            1.6.1
torch-geometric          2.4.0
torch-scatter            2.1.2
torch-sparse             0.6.18
torch-spline-conv        1.2.2
geoopt 0.5.0
```
<a name="instructions"/>

## 3. Examples

#### 3.1 Run HGCN on Cora dataset
`bash examples/icml2023/hgnn/HGCN/batch_cora_8.sh`
\
results: 0.7880, 0.8230, 0.8310

#### 3.2 Run HGCN on Citeseer dataset
`bash examples/icml2023/hgnn/HGCN/batch_citeseer_8.sh`
\
results: 0.6250, 0.7360, 0.7230

#### 3.3 Run HGCN on PubMed dataset
`bash examples/icml2023/hgnn/HGCN/batch_pubmed_8.sh`
\
results: 0.7780, 0.7990, 0.7900

#### 3.4 Run HGCN on Airport dataset
`bash examples/icml2023/hgnn/HGCN/batch_disease_256.sh`
\
results: 0.9389, 0.9427, 0.9447

#### 3.4 Run HGCN on Disease dataset
`bash examples/icml2023/hgnn/HGCN/batch_disease_8.sh`
\
results: 0.8936, 0.8696, 0.9318

#### 3.5 Run HGCN on Disease dataset
`bash examples/icml2023/hgnn/HGCN/batch_disease_256.sh`
\
results: 0.9474, 0.9328, 0.9569

Please check the folder of examples for more usages of the codes.
<a name="citation"/>

## 4. Citation

If you find this code useful in your research, please cite the following paper:

@inproceedings{yang2023hyperbolic,
author = {Yang, Menglin and Zhou, Min and Ying, Rex and Chen, Yankai and King, Irwin},
title = {Hyperbolic Representation Learning: Revisiting and Advancing},
year = {2023},
booktitle = {Proceedings of the 40th International Conference on Machine Learning},
articleno = {1654},
numpages = {21},
location = {Honolulu, Hawaii, USA},
series = {ICML'23}
}



