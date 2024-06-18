# A Coding-Theoretic Analysis of Hyperspherical Prototypical Learning Geometry

This repository provides an example implementation for the paper **A Coding-Theoretic Analysis of Hyperspherical Prototypical Learning Geometry** by Martin Lindström, Borja Rodríguez Gálvez, Ragnar Thobaben, and Mikael Skoglund at [KTH Royal Institute of Technology](https://www.kth.se), Stockholm, Sweden.

In the paper, we provide an analysis of the geometry in hyperspherical prototypical learning, a form of supervised representation learning on the hypersphere. Specifically, we analyse existing schemes to place well separated class prototypes on the hypersphere, thereby inducing class separation as an inductive bias. 

We propose two new methods for designing hypershperical prototypes and present sharp bounds on the optimal separation that can be achieved by placing an arbitrary number of prototypes $K ≤ 2n$ on a hypershpere of dimension $n$. Our approach rests on theory and concepts from error correcting codes, and our contributions are threefold:

1. We provide a new design approach for hyperspherical prototypes that maps binary linear codes defined over the $n$-dimensional Hamming space onto the ndimensional hypersphere $\mathbb{S}^{n−1}$. Our approach provides guarantees on the class separation by design, at the same time that it enables a more flexible trade-off between separation and the dimension $n$ for a given number of classes $K$.
2. We derive a *converse* bound on the guaranteed minimum prototype separation as well as an *achievable* bound that certifies that well-separated code-based prototypes exist. These bounds imply that for a large number of classes $K$ and in high dimensions $n$, the worstcase cosine similarity converges to zero. The bounds also show that our code-based prototypes closely approach optimal separation for $n \approx K/2$.
3. Finally, we provide alternative optimization-based hyperspherical prototypes which achieve the converse bound through a convex relaxation. These improve on the prototypes from literature, which do not achieve the converse bound.

Please consider citing this work using the BibTeX entry below.

```
to be added.
```

## Running the Code

Start by creating a configuration `.yaml` file with run parameters (such as optimiser, learning rate, etc.). Please see the example `.yaml` files for guidance on possible options. The code is then run with the command

    python3 main.py --identifier [identifier] --root [/path/to/saveroot] --config [/path/to/config] --dataset [/path/to/dataset]

where the `--root` flag denotes the save directory where logs, checkpoints, and final models will be saved; `--config` takes the path to the config `.yaml` file; and `--dataset` takes the root path to the dataset you wish to train on. The identifier has to be unique (since the subdirectory `/path/to/saveroot/identifier/` is created when starting the script, and old runs will not be overwritten).

## Recreating Figures

We also provide the code which generates the plots in the paper. These scripts can also serve as an introduction on how to use the prototype generating classes in `models.py`. The motivating example in Figure 1 can be generated with `generate_motivating_example.py`. Comparing the separation across different dimensions, like in Figures 2, 6, and 7, can be done with `compare_prototype_separation.py`. Finally, the cosine similarity histograms in Figures 3, 8, and 9 can be generated with `generate_prototype_histograms.py`.

## Dependencies

The code is written for PyTorch, and requires the usual PyTorch/Torchvision/NumPy family of packages, as well as a few from the Python3 standard library. It is written for PyTorch 2.3 and Torchvision 0.18, but would probably work for older versions as well (with minor, if any, modifications). Additionally, the code requires the `Galois` package (please find its documentation [here](https://galois.readthedocs.io/en/stable/)). One known issue with older versions of Torchvision is that the API for data import/augmentation changed with `torchvision.transforms.v2`, but changing only a couple of lines makes the code compatible with the older `torchvision.transforms` functions. In particular, the code is tested with the following versions:

| Package | Version | 
|--------------|:-----:| 
| NumPy | 1.24.4 | 
| Matplotlib | 3.8.2 |
| Pandas | 1.5.3 |
| SciPy | 1.12.0 |
| PyTorch | 2.3.0 | 
| Torchvision | 1.18.0 | 
| Tensorboard | 2.9.0 | 
| PyYaml | 6.0.1 |
| Galois | 0.3.8 |

## License

We make our code freely available under the MIT license, see the `LICENSE` file wherever applicable.

We greatly acknowledge the code by Kasarla et al. (2022) [through their GitHub repo](https://github.com/tkasarla/max-separation-as-inductive-bias/blob/main/LT_CIFAR/models/resnet.py): in particular, the implementation of ResNet-34 (i.e., the file `resnet.py`), and their prototype generation scheme (the `KasarlaCode` class in the `models.py` file).
