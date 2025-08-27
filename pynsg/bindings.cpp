#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include "efanna2e/index_nsg.h"
#include "efanna2e/util.h"

namespace py = pybind11;

class NSGWrapper {
private:
    efanna2e::IndexNSG* index;
    Py_ssize_t dimension;
    Py_ssize_t aligned_dimension;
    size_t n_points;

    float* align_data(py::array_t<float> data, size_t& aligned_dim)
    {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2 || buf.shape[1] != dimension)
        {
            throw std::runtime_error("Data must be 2D array with correct dimension");
        }

        size_t num_points = buf.shape[0];
        float* input_data = static_cast<float*>(buf.ptr);

        // the input is freed by data_align so we need to create a copy.
        float* data_copy = new float[num_points * dimension];
        std::memcpy(data_copy, input_data, num_points * dimension * sizeof(float));

        unsigned temp_dim = dimension;
        float* aligned = efanna2e::data_align(data_copy, num_points, temp_dim);
        aligned_dim = temp_dim;

        return aligned;
    }

public:
    NSGWrapper(size_t dim, size_t num_points, efanna2e::Metric metric = efanna2e::L2)
            : dimension(dim), n_points(num_points)
    {
        aligned_dimension = (dimension + 7) / 8 * 8;
        index = new efanna2e::IndexNSG(aligned_dimension, n_points, metric, nullptr);
    }

    ~NSGWrapper()
    {
        delete index;
    }

    void build_index(py::array_t<float> data, const std::string& knng_path,
            int L, int R, int C)
    {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2 || buf.shape[1] != dimension)
        {
            throw std::runtime_error("Data must be 2D array with correct dimension");
        }

        n_points = buf.shape[0];

        size_t temp_aligned_dim;
        float* aligned_data = align_data(data, temp_aligned_dim);
        aligned_dimension = temp_aligned_dim;

        efanna2e::Parameters params;
        params.Set<int>("L", L);
        params.Set<int>("R", R);
        params.Set<int>("C", C);
        params.Set<std::string>("nn_graph_path", knng_path);

        index->Build(n_points, aligned_data, params);
    }

    void optimize_graph(py::array_t<float, py::array::c_style | py::array::forcecast> data)
    {
        py::buffer_info buf = data.request();
        if (buf.ndim != 2 || buf.shape[1] != dimension)
        {
            throw std::runtime_error("Data must be 2D array with correct dimension");
        }

        size_t temp_aligned_dim;
        float* temp_aligned = align_data(data, temp_aligned_dim);

        std::unique_ptr<float, void(*)(float*)> aligned_guard(temp_aligned, [](float* ptr) {
#ifdef __APPLE__
          delete[] ptr;
#else
          free(ptr);
#endif
        });

        index->OptimizeGraph(temp_aligned);
    }

    std::vector<std::vector<int>> search_opt(py::array_t<float> queries, int k, int search_L)
    {
        py::buffer_info buf = queries.request();
        if (buf.ndim != 2 || buf.shape[1] != dimension)
        {
            throw std::runtime_error("Queries must be 2D array with correct dimension");
        }

        size_t n_queries = buf.shape[0];

        size_t temp_aligned_dim;
        float* aligned_queries = align_data(queries, temp_aligned_dim);

        std::unique_ptr<float, void(*)(float*)> query_guard(aligned_queries, [](float* ptr) {
#ifdef __APPLE__
          delete[] ptr;
#else
          free(ptr);
#endif
        });

        std::vector<std::vector<int>> results(n_queries, std::vector<int>(k));

        efanna2e::Parameters search_params;
        search_params.Set<int>("L_search", search_L);
        search_params.Set<unsigned>("P_search", search_L);

        for (size_t i = 0; i < n_queries; i++)
        {
            std::vector<unsigned> indices(k);
            index->SearchWithOptGraph(aligned_queries + i * temp_aligned_dim, k,
                    search_params, indices.data());
            for (int j = 0; j < k; j++)
            {
                results[i][j] = static_cast<int>(indices[j]);
            }
        }
        return results;
    }

    std::vector<std::vector<int>> search(py::array_t<float> queries, py::array_t<float> data, int k, int search_L)
    {
        py::buffer_info buf = queries.request();
        if (buf.ndim != 2 || buf.shape[1] != dimension)
        {
            throw std::runtime_error("Queries must be 2D array with correct dimension");
        }

        size_t n_queries = buf.shape[0];

        size_t queries_aligned_dim;
        float* aligned_queries = align_data(queries, queries_aligned_dim);
        size_t data_aligned_dim;
        float* aligned_data = align_data(data, data_aligned_dim);

        std::unique_ptr<float, void(*)(float*)> query_guard(aligned_queries, [](float* ptr) {
#ifdef __APPLE__
          delete[] ptr;
#else
          free(ptr);
#endif
        });

        std::unique_ptr<float, void(*)(float*)> data_guard(aligned_data, [](float* ptr) {
#ifdef __APPLE__
          delete[] ptr;
#else
          free(ptr);
#endif
        });

        std::vector<std::vector<int>> results(n_queries, std::vector<int>(k));

        efanna2e::Parameters search_params;
        search_params.Set<int>("L_search", search_L);
        search_params.Set<unsigned>("P_search", search_L);

        for (size_t i = 0; i < n_queries; i++)
        {
            std::vector<unsigned> indices(k);
            index->Search(aligned_queries + i * queries_aligned_dim,
                    aligned_data, k, search_params, indices.data());
            for (int j = 0; j < k; j++)
            {
                results[i][j] = static_cast<int>(indices[j]);
            }
        }
        return results;
    }

    void save_index(const std::string& path)
    {
        index->Save(path.c_str());
    }

    void load_index(const std::string& path)
    {
        index->Load(path.c_str());
    }
};

PYBIND11_MODULE(_bindings, m) {
    m.doc() = "NSG Python Wrapper";
    py::enum_<efanna2e::Metric>(m, "Metric")
        .value("L2", efanna2e::L2);

    py::class_<NSGWrapper>(m, "NSG")
        .def(py::init<size_t, size_t, efanna2e::Metric>(),
            py::arg("dimension"),
	    py::arg("num_points"),
            py::arg("metric") = efanna2e::L2)
        .def("build_index", &NSGWrapper::build_index,
            py::arg("data"),
            py::arg("knng_path"),
            py::arg("L"),
            py::arg("R"),
            py::arg("C"))
        .def("optimize_graph", &NSGWrapper::optimize_graph,
            py::arg("data"))
        .def("search_opt", &NSGWrapper::search_opt,
            py::arg("queries"),
            py::arg("k"),
            py::arg("search_L"))
        .def("search", &NSGWrapper::search,
            py::arg("queries"),
	    py::arg("data"),
            py::arg("k"),
            py::arg("search_L"))
        .def("save_index", &NSGWrapper::save_index,
            py::arg("path_to_index"))
        .def("load_index", &NSGWrapper::load_index,
            py::arg("path_to_index"));
}
