# NeuroSAT

A pytorch implementation of NeuraSAT([github](https://github.com/dselsam/neurosat), [paper](https://arxiv.org/abs/1802.03685))

In this implementation, we use SR(U(10, 40)) for training and SR(40) for testing, achieving the same accuracy 85% as in the original paper. The model was trained on a single K40 gpu for ~3 days following the parameters in the original paper.
