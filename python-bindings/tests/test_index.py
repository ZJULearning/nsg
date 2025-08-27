import numpy as np
import pytest
import faiss
from nsg_python import NSG, Metric

SIFT_BASE_PATH = "python-bindings/tests/data/sift/sift_base.fvecs"
SIFT_QUERY_PATH = "python-bindings/tests/data/sift/sift_query.fvecs"
SIFT_KNN_GRAPH_PATH = "python-bindings/tests/data/test200.graph"

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
    faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    base, queries = sift_data
    dim = base.shape[1]

    nsg = NSG(dimension=dim, num_points=len(base), metric=Metric.FAST_L2)
    nsg.build_index(base, SIFT_KNN_GRAPH_PATH, L=40, R=50, C=500)
    nsg.optimize_graph(base)

    k = 10
    test_queries = queries[:5]
    nsg_results = nsg.search_opt(test_queries, k, search_L=100)

    # Create Faiss FlatL2 index for ground truth
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(base.astype(np.float32))
    
    # Get ground truth results
    _, ground_truth_ids = faiss_index.search(test_queries.astype(np.float32), k)

    # Calculate recall for each query
    total_recall = 0.0
    for i, (nsg_result, gt_result) in enumerate(zip(nsg_results, ground_truth_ids)):
        nsg_set = set(nsg_result)
        gt_set = set(gt_result)
        
        recall = len(nsg_set.intersection(gt_set)) / len(gt_set)
        total_recall += recall
        
        print(f"Query {i}: Recall = {recall:.4f}")
    
    average_recall = total_recall / len(test_queries)
    print(f"Average Recall@{k}: {average_recall:.4f}")

    assert len(nsg_results) == 5
    assert all(len(row) == k for row in nsg_results)
