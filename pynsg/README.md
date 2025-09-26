# PyNSG

[![GitHub](https://img.shields.io/github/license/twuebker/nsg)](https://github.com/twuebker/nsg/blob/master/LICENSE.lesser)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

Python bindings for **Navigating Spreading-Out Graph (NSG)** - a fast and memory-efficient approximate nearest neighbor search algorithm.

## About NSG

NSG is a graph-based approximate nearest neighbor search algorithm that provides excellent search performance with low memory overhead. This library provides Python bindings for the original C++ implementation.

**Original Paper**: *Fast Approximate Nearest Neighbor Search with Navigating Spreading-out Graph* by Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai.

## Credits

This package provides Python bindings for the original NSG implementation:
- **Original NSG Repository**: [ZJULearning/nsg](https://github.com/ZJULearning/nsg)
- **Original Authors**: Cong Fu, Chao Xiang, Changxu Wang, Deng Cai
- **Python Bindings**: Created to enable easy integration with Python-based machine learning workflows (such as the ANN benchmarks)

## Requirements
The CPP code requires g++, cmake, libboost-dev and libgoogle-perftools-dev, which can be installed via 
```sudo apt-get install g++ cmake libboost-dev libgoogle-perftools-dev```.
All Python dependencies will be installed automatically. 

### Hardware

The underlying CPP implementation of NSG requires both OpenMP and AVX2.

## Installation

The knn extension requires faiss but provides an easy means of generating a knn graph in python.

```bash
pip install pynsg
pip install pynsg[knn]
```

## Quick Start

```python
from pynsg import NSG, Metric

nsg = NSG(dimension=128, num_points=1000, metric=Metric.L2)

# Build the index (requires a k-NN graph file - see below)
nsg.build_index(data, "knn_graph.graph", L=40, R=50, C=500)
k = 10
results = nsg.search(queries, data, k, search_L=100)

# Save and load index
nsg.save_index("my_index.nsg")
nsg2 = NSG(128, 1000, Metric.L2)
nsg2.load_index("my_index.nsg")
```

## Optimized Search 
The normal search functions above are recommended for low memory scenarios. The latter search yields better performance.

```python
import numpy as np
from pynsg import NSG, Metric

nsg = NSG(dimension=base_data.shape[1], 
          num_points=len(base_data), 
          metric=Metric.L2)

nsg.build_index(base_data, "knn_graph.graph", L=40, R=50, C=500)
nsg.optimize_graph(base_data)
k = 10
results = nsg.search_opt(queries, k, search_L=100)
```
## API Reference

### NSG Class

```python
NSG(dimension, num_points, metric)
```

**Parameters:**
- `dimension` (int): Dimensionality of the vectors
- `num_points` (int): Number of points in the dataset
- `metric` (Metric): Distance metric (Currently, only L2 is supported. Note that cosine similarity produces ranking identical to L2 on normalized vectors)

**Methods:**
- `build_index(data, knn_graph_path, L, R, C)`: Build the NSG index
- `search(queries, data, k, search_L)`: Search for k nearest neighbors
- `search_opt(queries, k, search_L)`: Optimized search (requires optimize_graph)
- `optimize_graph(data)`: Optimize the graph structure for faster search
- `save_index(path)`: Save the index to disk
- `load_index(path)`: Load an index from disk

### Metrics

Available distance metrics:
- `Metric.L2`: Standard L2 (Euclidean) distance
> [!NOTE]
> While cosine similarity is not directly exposed, it can be computed by first normalizing the vectors and then using L2.

## Generating k-NN Graphs

NSG requires an approximate k-NN graph written to a file in a specific format as input for building the index. There are a number of ways to obtain such a graph, for example using efanna_graph (recommended by the authors of the paper), which is only available in cpp. In python, you can use faiss, or another algorithm of your choice.
For convenience, if you install the extension of this package using `pip install pynsg[knn]`, faiss will be installed and you can use the function `create_graph_file` that uses faiss' hnsw index to quickly build an approximate knn graph. It uses OpenMP. 

This function is also exposed as a command line utility `nsg-build-knn`.

```python
from pynsg import create_graph_file

create_graph_file(filename="test200.graph", x=X, k=200, hnsw_efConstruction=200, hnsw_efSearch=128, hnsw_M=32, use_omp=True) 
```

## License

PyNSG is MIT-licensed.
The original NSG algorithm and implementation are credited to the authors of the ZJULearning/nsg repository.

## Citation

If you use this library in your research, please cite the original NSG paper:

```bibtex
@article{FuNSG17,
  author    = {Cong Fu and Chao Xiang and Changxu Wang and Deng Cai},
  title     = {Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graphs},
  journal   = {{PVLDB}},
  volume    = {12},
  number    = {5},
  pages     = {461 - 474},
  year      = {2019},
  url       = {http://www.vldb.org/pvldb/vol12/p461-fu.pdf},
  doi       = {10.14778/3303753.3303754}
}
```
