## 1. Motivation

This repository contains the code for the implementation of the node regression tasks on dynamic temporal graphs.

We tend to perform comparative analysis on different Temporal graph neural network (TGNNs) models using the England Covid 19 dataset. 

## 2. Environment

- Python 3.8.0
- PyTorch 2.0.*
- CUDA 11.7
- PyTorch Geometric Temporal (ref: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

Prepare the anaconda environment:

```bash
conda create -n tg python=3.8.0
conda activate tg
pip install -r requirements.txt
```
## References
[1] https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/index.html

[2] https://github.com/pyg-team/pytorch_geometric/issues/3230 

[3] Fey, Matthias, and Jan Eric Lenssen. "Fast graph representation learning with PyTorch Geometric." arXiv preprint arXiv:1903.02428 (2019).

[4] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[5] Pareja, Aldo, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao Schardl, and Charles Leiserson. "Evolvegcn: Evolving graph convolutional networks for dynamic graphs." In Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 04, pp. 5363-5370. 2020 


