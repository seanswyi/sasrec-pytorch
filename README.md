# SASRec PyTorch Implementation
_Official Paper: [Self-Attentive Sequential Recommendation (Kang and McAuley, 2018)](https://arxiv.org/abs/1808.09781)_ \
_Official TensorFlow Code: [kang205/SASRec](https://github.com/kang205/SASRec)_ \
_Official PyTorch Code: [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)_

Personal implementation of the SASRec model in PyTorch. There are some changes that I've made compared to the official implementations but the overall model, logic, pipeline, etc. are largely the same.

There are two versions of implementation that I tried out: one using my own self-attention implementation (inspired by my previous Transformer implementation: [seanswyi/transformer-implementation](https://github.com/seanswyi/transformer-implementation)) and the other using [PyTorch's official Multi-Head Attention implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). Extensive experiments showed that there's not that big of a difference between the two, so I opted to use PyTorch's implementation for the sake of reliability.

---

## Results

Experiments were conducted on the four datasets used in the original paper: Amazon Beauty, Amazon Games, Steam, and MovieLens 1M. Hyperparameters are set to be the same, except for setting the number of epochs for early stopping to 100 rather than 20.

Results differ slightly from the officially reported results, and evaluation is also conducted differently. The official implementations conduct evaluation on both the validation and test sets at every evaluation step, but my implementation only conducts evaluation on the validation set and uses the test set only once at the end of training with the best performing checkpoint.
