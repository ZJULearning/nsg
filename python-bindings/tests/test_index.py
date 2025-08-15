import numpy as np
import pytest
from nsg_python import NSG, Metric

SIFT_BASE_PATH = "python-bindings/tests/data/sift/sift_base.fvecs"
SIFT_QUERY_PATH = "python-bindings/tests/data/sift/sift_query.fvecs"
SIFT_KNN_GRAPH_PATH = "python-bindings/tests/data/sift_200nn.graph"

def read_fvecs(fname):
    """Load fvecs file into numpy array"""
    a = np.fromfile(fname, dtype=np.int32)
    if a.size == 0:
        raise FileNotFoundError(f"File {fname} is empty or not found")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].astype(np.float32)

@pytest.fixture(scope="module")
def sift_data():
    base = read_fvecs(SIFT_BASE_PATH)
    queries = read_fvecs(SIFT_QUERY_PATH)
    return base, queries

def test_build_and_search(sift_data, tmp_path):
    base, queries = sift_data
    dim = base.shape[1]

    nsg = NSG(dimension=dim, num_points=len(base), metric=Metric.L2)
    nsg.build_index(base, SIFT_KNN_GRAPH_PATH, L=10, R=10, C=10)

    k = 10
    results = nsg.search(queries[:5], base, k, search_L=100)
    assert len(results) == 5
    assert all(len(row) == k for row in results)

    save_path = tmp_path / "index.nsg"
    nsg.save_index(str(save_path))

    nsg2 = NSG(dim, len(base), Metric.L2)
    nsg2.load_index(str(save_path))

    results2 = nsg2.search(queries[:5], base, k, search_L=100)
    assert len(results2) == 5
    assert all(len(row) == k for row in results2)

def test_build_and_search_opt(sift_data):
    base, queries = sift_data
    dim = base.shape[1]
    
    nsg = NSG(dimension=dim, num_points=len(base), metric=Metric.L2)
    nsg.build_index(base, SIFT_KNN_GRAPH_PATH, L=10, R=10, C=10)
    nsg.optimize_graph(base)
    
    k = 10
    results = nsg.search_opt(queries[:5], k, search_L=100) 
    
    assert len(results) == 5
    assert all(len(row) == k for row in results)
