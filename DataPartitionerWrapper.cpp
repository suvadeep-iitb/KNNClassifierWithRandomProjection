#include <iostream>
#include <list>
#include <vector>
#include <set>
#include <utility>
#include <tuple>
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>
#include "DataPartitioner.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

using namespace std;


inline np::ndarray c2ndarray_int32(int32_t* carray, int n) 
{
  np::dtype dt = np::dtype::get_builtin<int32_t>();
  p::tuple shape = p::make_tuple(n);
  p::tuple stride = p::make_tuple(sizeof(int32_t));
  p::object own;

  return np::from_data(carray, dt, shape, stride, own);
}


inline np::ndarray c2ndarray_float(float* carray, int n) 
{
  np::dtype dt = np::dtype::get_builtin<float>();
  p::tuple shape = p::make_tuple(n);
  p::tuple stride = p::make_tuple(sizeof(float));
  p::object own;

  return np::from_data(carray, dt, shape, stride, own);
}


void csr_sparse2vec(np::ndarray const & indices,
                    np::ndarray const & data,
                    np::ndarray const & indptr,
                    vector<vector<pair<int, float>>> & data_vec)
{
  size_t nrow = indptr.shape(0) - 1;
  size_t data_size = data.shape(0);
  bool is_float = (data.get_dtype() == np::dtype::get_builtin<float>());
  bool is_double = (data.get_dtype() == np::dtype::get_builtin<double>());

  int32_t* indices_ptr = reinterpret_cast<int32_t*>(indices.get_data());
  int32_t* indptr_ptr = reinterpret_cast<int32_t*>(indptr.get_data());
  float* data_ptr_float = NULL;
  double* data_ptr_double = NULL;
  if (is_float) {
    data_ptr_float = reinterpret_cast<float*>(data.get_data());
  }
  else if (is_double) {
    data_ptr_double = reinterpret_cast<double*>(data.get_data());
  }
  else
    assert(0);

  data_vec.clear();
  data_vec.resize(nrow); 
  for (auto r = 0; r < nrow; r++) {
    data_vec[r].resize(indptr_ptr[r+1]-indptr_ptr[r]);
    for (auto c = indptr_ptr[r]; c < indptr_ptr[r+1]; c++) {
      data_vec[r][c-indptr_ptr[r]].first = indices_ptr[c];
      if (is_float) {
        data_vec[r][c-indptr_ptr[r]].second = data_ptr_float[c];
      }
      else {
        data_vec[r][c-indptr_ptr[r]].second = data_ptr_double[c];
      }
    }
  }
}


void csr_sparse2vec(np::ndarray const & indices,
                    np::ndarray const & indptr,
                    vector<vector<int>> & data_vec)
{
  size_t nrow = indptr.shape(0) - 1;

  int32_t* indices_ptr = reinterpret_cast<int32_t*>(indices.get_data());
  int32_t* indptr_ptr = reinterpret_cast<int32_t*>(indptr.get_data());

  data_vec.clear();
  data_vec.resize(nrow);
  for (auto r = 0; r < nrow; r++) {
    data_vec[r].resize(indptr_ptr[r+1]-indptr_ptr[r]);
    for (auto c = indptr_ptr[r]; c < indptr_ptr[r+1]; c++) {
      data_vec[r][c-indptr_ptr[r]] = indices_ptr[c];
    }
  }
}


void ndarray2vec(np::ndarray const & data,
                 vector<vector<pair<int, float>>> & vec)
{
  size_t nrow = data.shape(0);
  size_t ncol = data.shape(1);
  bool is_float = (data.get_dtype() == np::dtype::get_builtin<float>());
  float* data_ptr_float = NULL;
  double* data_ptr_double = NULL;
  if (is_float)
    data_ptr_float = reinterpret_cast<float*>(data.get_data());
  else
    data_ptr_double = reinterpret_cast<double*>(data.get_data());
  vec.resize(ncol);
  for (size_t r = 0; r < nrow; r++) {
    for (size_t c = 0; c < ncol; c++) {
      int k = r;
      float d;
      if (is_float) {
          d = (float)data_ptr_float[r*ncol+c];
      }
      else {
          d = (float)data_ptr_double[r*ncol+c];
      }
      if (d > 0.0)
        vec[c].resize(vec[c].size()+1);
        vec[c].back().first = k;
        vec[c].back().second = d;
    }    
  }

  /////
  cout << "1 Centers" << endl;
  for (auto r = 0; r < vec.size(); r++) {
    for (auto c = 0; c < vec[r].size(); c++)
      cout << vec[r][c].first << ":" << vec[r][c].second << " ";
    cout << endl;
  }
  /////
}


void csr_sparse2vec(size_t r, 
                    np::ndarray const & indices, 
                    np::ndarray const & data, 
                    np::ndarray const & indptr, 
                    vector<pair<int, float>> & vec)
{
  size_t nrow = indptr.shape(0) - 1;
  bool is_float = (data.get_dtype() == np::dtype::get_builtin<float>());
  bool is_double = (data.get_dtype() == np::dtype::get_builtin<double>());

  int32_t* indices_ptr = reinterpret_cast<int32_t*>(indices.get_data());
  int32_t* indptr_ptr = reinterpret_cast<int32_t*>(indptr.get_data());
  float* data_ptr_float = NULL;
  double* data_ptr_double = NULL;
  if (is_float) {
    data_ptr_float = reinterpret_cast<float*>(data.get_data());
  }
  else if (is_double) {
    data_ptr_double = reinterpret_cast<double*>(data.get_data());
  }
  else
    assert(0);

  vec.clear();
  vec.resize(indptr_ptr[r+1]-indptr_ptr[r]);
  for (auto c = indptr_ptr[r]; c < indptr_ptr[r+1]; c++) {
    vec[c-indptr_ptr[r]].first = indices_ptr[c];
    if (is_float) {
      vec[c-indptr_ptr[r]].second = data_ptr_float[c];
    }
    else {
      vec[c-indptr_ptr[r]].second = data_ptr_double[c];
    }
 }
}

/*
void vec2csr_sparse(vector<vector<pair<int, float>>> const & data_vec,
                    np::ndarray & indices,
                    np::ndarray & data,
                    np::ndarray & indptr)
{
  int nrow = data_vec.size();
  int nnz = 0;
  for (auto&& pair_vec: data_vec) {
    nnz += pair_vec.size();
  }

  int32_t* indices_ptr = (int32_t*)malloc(nnz*sizeof(int32_t));
  float* data_ptr = (float*)malloc(nnz*sizeof(float));
  int32_t* indptr_ptr = (int32_t*)malloc((nrow+1)*sizeof(int32_t));

  indptr_ptr[0] = 0;
  for (auto r = 0; r < nrow; r++) {
    auto s_idx = indptr_ptr[r];
    indptr_ptr[r+1] = s_idx + data_vec[r].size();
    for (auto c = 0; c < data_vec[r].size(); c++) {
      indices_ptr[s_idx+c] = data_vec[r][c].first;
      data_ptr[s_idx+c] = data_vec[r][c].second;
    }
  }

  indices = c2ndarray_int32(indices_ptr, nnz);
  indptr = c2ndarray_int32(indptr_ptr, nrow+1);
  data = c2ndarray_float(data_ptr, nnz);
}
*/


class DataPartitionerWrapper
{
  private:
    DataPartitioner dataPartitioner_;

  public:
    DataPartitionerWrapper() 
    {}

    DataPartitionerWrapper(DataPartitioner dataPartitioner) :
                           dataPartitioner_(dataPartitioner)
    {}

    /*
    DataPartitionerWrapper(np::ndarray const & centers)
    {
      size_t K = centers.shape(0);
      vector<vector<pair<int, float>>> w_index; 
      ndarray2vec(centers, w_index);
      /////
      cout << "2 Centers" << endl;
      for (auto r = 0; r < w_index.size(); r++) {
        for (auto c = 0; c < w_index[r].size(); c++)
          cout << w_index[r][c].first << ":" << w_index[r][c].second << " ";
        cout << endl;
      }
      /////
      dataPartitioner_ = DataPartitioner(K, w_index);
      w_index = dataPartitioner_.w_index();
      /////
      cout << "3 Centers" << endl;
      for (auto r = 0; r < w_index.size(); r++) {
        for (auto c = 0; c < w_index[r].size(); c++)
          cout << w_index[r][c].first << ":" << w_index[r][c].second << " ";
        cout << endl;
      }
      /////
    }
    */

    float RunPairwise( np::ndarray const & feature_indices,
                              np::ndarray const & feature_data,
                              np::ndarray const & feature_indptr,
                              np::ndarray const & label_indices,
                              np::ndarray const & label_indptr,
                              size_t K, size_t max_iter,
                              size_t num_nn, int label_normalize,
                              float eta0, float lambda, float gamma, 
                              int seed, int verbose)
    {
      assert(feature_indptr.shape(0) == label_indptr.shape(0));
      assert(feature_data.shape(0) == feature_indices.shape(0));

      vector<vector<pair<int, float>>> feature_vec;
      csr_sparse2vec(feature_indices, feature_data, feature_indptr, feature_vec);

      vector<vector<int>> label_vec;
      csr_sparse2vec(label_indices, label_indptr, label_vec);

      dataPartitioner_.Clear();
      return dataPartitioner_.RunPairwise(feature_vec, label_vec,
                                          K, max_iter, num_nn,
                                          label_normalize, eta0,
                                          lambda, gamma, seed,
                                          verbose);
    }

    float RunNeighbourExpansionEP(np::ndarray const & label_indices,
                               np::ndarray const & label_indptr,
                               np::ndarray & partitions,
                               size_t K, size_t num_nn, int label_normalize,
                               float replication_factor, int seed, int verbose)
    {
      vector<vector<int> > label_vec;
      vector<set<size_t> > cluster_assign;
      csr_sparse2vec(label_indices, label_indptr, label_vec);

      dataPartitioner_.Clear();
      float rep_factor = dataPartitioner_.RunNeighbourExpansionEP(label_vec, cluster_assign,
                                             K, num_nn, label_normalize,
                                             replication_factor, seed,
                                             verbose);
      label_vec.clear();

      assert(partitions.get_nd() == 2);
      assert(partitions.shape(0) == K);
      assert(partitions.shape(1) == label_indptr.shape(0)-1);
      bool is_float = (partitions.get_dtype() == np::dtype::get_builtin<float>());
      bool is_double = (partitions.get_dtype() == np::dtype::get_builtin<double>());
    
      float* partitions_ptr_float = NULL; 
      double* partitions_ptr_double = NULL; 
      if (is_float)
        partitions_ptr_float = reinterpret_cast<float*>(partitions.get_data());
      else if (is_double)
        partitions_ptr_double = reinterpret_cast<double*>(partitions.get_data());
      else
        assert(0);

      size_t nrow = partitions.shape(0);
      size_t ncol = partitions.shape(1);
      for (auto r = 0; r < nrow; r++) {
        for (auto c = 0; c < ncol; c++) {
          if (is_float)
            partitions_ptr_float[r*ncol+c] = (float)0.0;
          else
            partitions_ptr_double[r*ncol+c] = (double)0.0;
        }
      }

      assert(cluster_assign.size() == nrow);
      for (auto r = 0; r < cluster_assign.size(); r++) {
        for (auto k: cluster_assign[r]) {
          assert(k < ncol);
          if (is_float)
            partitions_ptr_float[r*ncol+k] = (float)1.0;
          else
            partitions_ptr_double[r*ncol+k] = (double)1.0;
        }
      }

      return rep_factor;
    }

    void GetNearestClusters(np::ndarray const & feature_indices,
                                  np::ndarray const & feature_data,
                                  np::ndarray const & feature_indptr,
                                  np::ndarray & assignments)
    {
      assert(feature_indptr.shape(0)-1 == assignments.shape(0));
      assert(feature_data.shape(0) == feature_indices.shape(0));
      //assert(assignments.get_dtype() == np::dtype::get_builtin<int32_t>);

      int32_t* assign_ptr = reinterpret_cast<int32_t*>(assignments.get_data());
      size_t nrow = feature_indptr.shape(0) - 1;
      for (auto r = 0; r < nrow; r++) {
        vector<pair<int, float>> row_vec;
        csr_sparse2vec(r, feature_indices, feature_data, feature_indptr, row_vec);

        assign_ptr[r] = dataPartitioner_.GetNearestCluster(row_vec);
      }
    }

    size_t GetK() { return dataPartitioner_.K(); }

    void GetCenters(const np::ndarray & centers)
    {
      assert(centers.get_nd() == 2);
      assert(centers.shape(0) == dataPartitioner_.K());
      bool is_float = (centers.get_dtype() == np::dtype::get_builtin<float>());
      bool is_double = (centers.get_dtype() == np::dtype::get_builtin<double>());
    
      float* centers_ptr_float = NULL; 
      double* centers_ptr_double = NULL; 
      if (is_float)
        centers_ptr_float = reinterpret_cast<float*>(centers.get_data());
      else if (is_double)
        centers_ptr_double = reinterpret_cast<double*>(centers.get_data());
      else
        assert(0);

      size_t nrow = centers.shape(0);
      size_t ncol = centers.shape(1);
      for (auto r = 0; r < nrow; r++) {
        for (auto c = 0; c < ncol; c++) {
          if (is_float)
            centers_ptr_float[r*ncol+c] = (float)0.0;
          else
            centers_ptr_double[r*ncol+c] = (double)0.0;
        }
      }

      vector<vector<pair<int, float>>> w_index = dataPartitioner_.w_index();
      assert(w_index.size() <= ncol);
      for (auto r = 0; r < w_index.size(); r++) {
        for (auto c = 0; c < w_index[r].size(); c++) {
          int k = w_index[r][c].first;
          assert(k < ncol);
          if (is_float)
            centers_ptr_float[k*ncol+r] = (float)w_index[r][c].second;
          else
            centers_ptr_double[k*ncol+r] = (double)w_index[r][c].second;
        }
      }
    }

    void Clear() {
      dataPartitioner_.Clear();
    }
};


    


BOOST_PYTHON_MODULE(data_partitioner)
{
  np::initialize();
 
  using namespace p; 
  class_<DataPartitionerWrapper>("DataPartitioner")
          .def(init<DataPartitioner>())
          //.def(init<np::ndarray const &>())
          .def("RunPairwise", &DataPartitionerWrapper::RunPairwise)
          .def("GetNearestClusters", &DataPartitionerWrapper::GetNearestClusters)
          .def("RunNeighbourExpansionEP", &DataPartitionerWrapper::RunNeighbourExpansionEP)
          .def("GetK", &DataPartitionerWrapper::GetK)
          .def("GetCenters", &DataPartitionerWrapper::GetCenters)
          .def("Clear", &DataPartitionerWrapper::Clear)
  ;
}

