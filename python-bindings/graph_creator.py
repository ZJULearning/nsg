from argparse import ArgumentParser
import faiss
import numpy as np

def create_graph_file(filename: str, x, k, use_omp=True):
    if use_omp:
        omp_threads = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(omp_threads)

    n, d = x.shape
    m = 32
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 128
    index.add(x)

    distances, neighbors = index.search(x, k + 1)

    for i in range(n):
        if neighbors[i, 0] != i:
            neighbors[i, -1] = i
    neighbors = neighbors[:, 1:]

    k = neighbors.shape[1]

    with open(filename, "wb") as f:
        f.write(np.uint32(k).tobytes())
        for i in range(n):
            f.write(np.uint32(i).tobytes())
            neighbors[i].astype(np.uint32).tofile(f)

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='nnGraphMaker',
        description='Creates approximate nn graph using HNSW.',
    )
    parser.add_argument('-f', '--filename', required=True,
                        help='Output filename for the graph.')
    parser.add_argument('-k', '--knn', type=int, required=True,
                        help='Number of nearest neighbors to search for.')
    parser.add_argument('-i', '--input', required=True,
                        help='Input vectors in fvecs file.')

    args = parser.parse_args()
    vecs = read_fvecs(args.input)
    create_graph_file(args.filename, vecs, args.knn)
