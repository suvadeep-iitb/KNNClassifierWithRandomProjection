//
// Copyright (C) 2017 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "DataPartitioner.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <set>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "Utils.h"
#include "Reindexer.h"

const size_t mat_bit_size = 24;

class HashMatrix {
  public:
    HashMatrix() {};
    HashMatrix(size_t rows, size_t cols, size_t bit_size) : rows_(rows), cols_(cols) {
      size_t cap = 1 << bit_size;
      mask_ = cap - 1;
      size_t size = std::min(cap, rows * cols);
      vals_.resize(size, 0.0f);
    };
    ~HashMatrix() {};
    void FilledWith(float v) { std::fill(vals_.begin(), vals_.end(), v); };
    float operator()(size_t row, size_t col) const { return vals_[(row * cols_ + col) & mask_]; }
    float& operator()(size_t row, size_t col)      { return vals_[(row * cols_ + col) & mask_]; }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

  private:
    size_t rows_;
    size_t cols_;
    size_t mask_;
    std::vector<float> vals_;
};

struct FindNode {
  FindNode(size_t nid) : nid(nid) {}

  bool operator()(std::pair<size_t, float> const & p)
  { return (p.first == nid);}

  size_t nid;
};

inline float sigmoid(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  } else {
    float e = std::exp(x);
    return e / (e + 1.0f);
  }
}

inline float sparse_dot(const std::vector<std::pair<int, float> > &x,
                        const HashMatrix &m, size_t row) {
  float v = 0.0f;
  for (size_t f = 0; f < x.size(); ++f) {
    size_t k = x[f].first;
    if (k >= m.cols()) { continue; }
    v += m(row, k) * x[f].second;
  }
  return v;
}

inline void add_vec_ftrl(const std::vector<std::pair<int, float> > &x,
                         float eta0, float lambda, float coef,
                         size_t row,
                         HashMatrix *wvec,
                         HashMatrix *gvec,
                         HashMatrix *zvec) {

  for (size_t f = 0; f < x.size(); ++f) {
    int k = x[f].first;
    float g = coef * x[f].second;
    float old_n = std::sqrt((*gvec)(row, k));
    (*gvec)(row, k) += g * g;
    float new_n = std::sqrt((*gvec)(row, k));
    float s = (new_n - old_n) / eta0;
    (*zvec)(row, k) += g + s * (*wvec)(row, k);
    float z = (*zvec)(row, k);

    if (std::fabs(z) <= lambda) {
      (*wvec)(row, k) = 0.0f;
    } else {
      z -= (z > 0.0f) ? lambda : -lambda;
      (*wvec)(row, k) = eta0 * z / new_n;
    }
  }
}

float get_nearest_centers(const std::vector<std::pair<int, float> > &datum,
                          const HashMatrix &cvec,
                          std::vector<size_t> *centers) {
  if (centers == NULL) { return 0.0f; }

  float max_ip = -FLT_MAX;
  for (size_t k = 0; k < cvec.rows(); ++k) {
    float ip = sparse_dot(datum, cvec, k);
    if (ip > max_ip) {
      centers->clear();
      centers->push_back(k);
      max_ip = ip;
    } else if (std::fabs(max_ip - ip) < FLT_EPSILON) {
      centers->push_back(k);
    }
  }

  return max_ip;
}

size_t update_cvec(
    const std::vector<std::vector<std::pair<int, float> > > &data_vec,
    size_t base_idx,
    const std::vector<std::pair<size_t, float> > &pos_vec,
    const std::vector<std::pair<size_t, float> > &neg_vec,
    float eta0, float lambda,
    std::mt19937 *rnd_gen,
    HashMatrix *cvec,
    HashMatrix *gvec,
    HashMatrix *zvec,
    std::vector<size_t> *cluster_vec) {

  if (pos_vec.size() == 0) { return 0; }

  size_t num_correct = 0;
  std::vector<size_t> centers;

  get_nearest_centers(data_vec[base_idx], *cvec, &centers);
  if (centers.size() > 1) { std::shuffle(centers.begin(), centers.end(), *rnd_gen); }
  size_t bc = centers[0];
  (*cluster_vec)[base_idx] = bc;

  for (size_t i = 0; i < pos_vec.size(); ++i) {
    size_t idx = pos_vec[i].first;
    float z = sparse_dot(data_vec[idx], *cvec, bc);
    float coef = 1.0f - sigmoid(z);
    add_vec_ftrl(data_vec[idx], eta0, lambda, coef, bc, cvec, gvec, zvec);
    if ((*cluster_vec)[idx] == bc) { ++num_correct; }
  }
  for (size_t i = 0; i < neg_vec.size(); ++i) {
    size_t idx = neg_vec[i].first;
    float z = sparse_dot(data_vec[idx], *cvec, bc);
    float coef = - sigmoid(z);
    add_vec_ftrl(data_vec[idx], eta0, lambda, coef, bc, cvec, gvec, zvec);
  }

  return num_correct;
}

void get_positives(
    const std::vector<std::vector<int> > &labels_vec,
    size_t num_pos, int label_normalize, size_t cost_per_sample, int verbose,
    std::vector<std::vector<std::pair<size_t, float> > > *pos_vec) {

  auto comp = [](const std::pair<size_t,float> &a, const std::pair<size_t,float> &b) {return a.second>b.second;};

  std::vector<size_t> indices(labels_vec.size());
  for (size_t i = 0; i < indices.size(); ++i) { indices[i] = i; }

  yj::xmlc::Reindexer l_idxer;
  std::vector<std::vector<int> > r_labels_vec;
  int max_lid = l_idxer.IndexLabels(labels_vec, indices, &r_labels_vec);

  // build inverted index of labels
  std::vector<std::vector<std::pair<size_t, float> > > l_inv_idx(max_lid + 1);
  for (size_t i = 0; i < r_labels_vec.size(); ++i) {
    float v = (label_normalize > 0) ? 1.0f / r_labels_vec[i].size() : 1.0f;
    for (size_t l = 0; l < r_labels_vec[i].size(); ++l) {
      l_inv_idx[r_labels_vec[i][l]].push_back(std::make_pair(i, v));
    }
  }
  if (verbose > 0) { fprintf(stderr, "build inverted index\n"); }

  std::vector<size_t> idx_len_vec(l_inv_idx.size());
  for (size_t i = 0; i < idx_len_vec.size(); ++i) { idx_len_vec[i] = l_inv_idx[i].size(); }
  std::sort(idx_len_vec.begin(), idx_len_vec.end(), std::greater<size_t>());

  // set thresh for eliminating head labels
  size_t budget = r_labels_vec.size() * cost_per_sample;
  size_t thresh = idx_len_vec[0];
  size_t count = 0;
  size_t thresh_idx = 0;
  for (size_t i = 0; i < idx_len_vec.size(); ++i) {
    size_t idx = idx_len_vec.size() - 1 - i;
    count += idx_len_vec[idx] * idx_len_vec[idx] - idx_len_vec[idx];
    if (count > budget) { break; }
    thresh = idx_len_vec[idx];
    thresh_idx = idx;
  }
  if (verbose > 0) {
    fprintf(stderr, "Top idx_len_vec -> ");
    for (size_t i = 0; i < std::min(10LU, idx_len_vec.size()); ++i) { fprintf(stderr, "%lu, ", idx_len_vec[i]); }
    fprintf(stderr, "\n");
    fprintf(stderr, "thresh: %lu, idx: %lu (in %d)\n", thresh, thresh_idx, max_lid);
  }

  std::vector<std::pair<size_t, float> > score_vec;
  std::unordered_map<size_t, float> score_map;

  pos_vec->resize(r_labels_vec.size());
  for (size_t i = 0; i < r_labels_vec.size(); ++i) {
    if (verbose > 0 && i % 1000 == 0) { fprintf(stderr, "\rsearch knn: %luK", i / 1000); }
    score_map.clear();

    for (size_t l = 0; l < r_labels_vec[i].size(); ++l) {
      size_t lid = r_labels_vec[i][l];
      size_t list_size = l_inv_idx[lid].size();
      if (list_size > thresh) { continue; } // skip too long list
      for (size_t j = 0; j < list_size; ++j) {
        size_t idx = l_inv_idx[lid][j].first;
        if (i == idx) { continue; }
        float v = l_inv_idx[lid][j].second;

        auto itr = score_map.find(idx);
        if (itr != score_map.end()) { itr->second += v; }
        else                        { score_map[idx] = v; }
        ++count;
      }
    }

    score_vec.clear();
    for (auto itr = score_map.begin(); itr != score_map.end(); ++itr) {
      if (score_vec.size() >= num_pos) {
        if (itr->second <= score_vec.front().second) { continue; }
        std::pop_heap(score_vec.begin(), score_vec.end(), comp);
        score_vec.pop_back();
      }
      score_vec.push_back(*itr);
      std::push_heap(score_vec.begin(), score_vec.end(), comp);
    }
    std::sort_heap(score_vec.begin(), score_vec.end(), comp);

    (*pos_vec)[i].clear();
    if (score_vec.size() > 0) { // if not sufficient, amplify positives
      for (size_t j = 0; j < num_pos; ++j) {
        (*pos_vec)[i].push_back(score_vec[j % score_vec.size()]);
      }
    }
  }
  if (verbose > 0) { fprintf(stderr, "\rget_positives done!\n"); }
}

void get_edge_set(
    std::vector<std::set<size_t> > &nn_graph, 
    std::set<size_t> &cluster_assignment, 
    std::set<size_t> &left_vertices, 
    float delta_min,
    float delta_max)
{
    std::set<size_t> S_minus_C;
    std::unordered_set<size_t> S;
    size_t edge_count = 0;

    while (edge_count <= delta_max) {
      std::set<size_t> diff;
      size_t x;
      if (S_minus_C.size() == 0) {
        if ((left_vertices.size() == 0) || (edge_count >= delta_min))
          return;
        size_t r = rand() % left_vertices.size();
        std::set<size_t>::const_iterator it(left_vertices.begin());
        std::advance(it, r);
        x = *it;
      }
      else {
        float min_val = nn_graph.size();
        for (auto&& it: S_minus_C) {
          size_t cur_node = it;
          float val = 0;
          for (auto&& nn: nn_graph[cur_node]) {
            if (S.find(nn) == S.end()) 
              val += 1;
          }

          if (val < min_val) {
            min_val = val;
            x = cur_node;
          }
        }
      }

      // AllocEdges
      if (S.find(x) == S.end())
        S.insert(x);
      else
        S_minus_C.erase(x);
      left_vertices.erase(x);
      std::set<size_t> yset = nn_graph[x];
      for (auto&& y: yset) {
        if (S.find(y) != S.end()) continue;

        S.insert(y);
        S_minus_C.insert(y);
        std::set<size_t> zset = nn_graph[y];
        for (auto z: zset) {
          if (S.find(z) == S.end()) continue;

          edge_count++;
          cluster_assignment.insert(y);
          cluster_assignment.insert(z);

          auto res = std::find(nn_graph[y].begin(), nn_graph[y].end(), z);
          if (res != nn_graph[y].end()) nn_graph[y].erase(res);

          res = std::find(nn_graph[z].begin(), nn_graph[z].end(), y);
          if (res != nn_graph[z].end()) nn_graph[z].erase(res);
        }
      }
    }
}

void get_edge_set(
    std::vector<std::vector<std::pair<size_t, float> > > &nn_graph, 
    std::set<size_t> &cluster_assignment, 
    std::set<size_t> &left_vertices, 
    float delta_min,
    float delta_max)
{
    std::set<size_t> C;
    std::set<size_t> S;
    size_t edge_count = 0;

    while (edge_count <= delta_max) {
      std::set<size_t> diff;
      std::set_difference(S.begin(), S.end(), C.begin(), C.end(),
                          std::inserter(diff, diff.begin()));
      size_t x;
      if (diff.size() == 0) {
        if ((left_vertices.size() == 0) || (edge_count >= delta_min))
          return;
        size_t r = rand() % left_vertices.size();
        std::set<size_t>::const_iterator it(left_vertices.begin());
        std::advance(it, r);
        x = *it;
      }
      else {
        float min_val = nn_graph.size();
        for (auto&& it: diff) {
          size_t cur_node = it;
          float val = 0;
          auto s = S.begin();
          auto p = nn_graph[cur_node].begin();
          while(p != nn_graph[cur_node].end()) {
            if (s == S.end()) break;
            if (p->first < *s) {
              val += p->second;
              p++;
            }
            else if (*s < p->first) { s++; }
            else {s++; p++;}
          }

          if (val < min_val) {
            min_val = val;
            x = cur_node;
          }
        }
      }

      // AllocEdges
      C.insert(x);
      S.insert(x);
      left_vertices.erase(x);
      std::set<size_t> yset;
      for (auto&& ity: nn_graph[x]) {
        size_t y = ity.first;
        if(std::find(S.begin(), S.end(), y) == S.end()) {
          S.insert(y);
          yset.insert(y);
        }
      }

      for (auto&& y: yset) {
        std::set<size_t> zset;
        for (auto&& itz: nn_graph[y]) zset.insert(itz.first);
        for (auto z: zset) {
          if (std::find(S.begin(), S.end(), z) == S.end())
            continue;
          
          edge_count++;
          cluster_assignment.insert(y);
          cluster_assignment.insert(z);
          auto res = std::find_if(nn_graph[y].begin(), nn_graph[y].end(),
                               FindNode(z));
          nn_graph[y].erase(res);
          res = std::find_if(nn_graph[z].begin(), nn_graph[z].end(),
                               FindNode(y));
          nn_graph[z].erase(res);
          if (edge_count > delta_max)
            return;
        }
      }
    }
}

void sampling_negatives_uniform(
    const std::vector<std::vector<int> > &labels_vec,
    size_t num_neg,
    std::mt19937 *rnd_gen,
    std::vector<std::pair<size_t, float> > *neg_vec) {

  std::uniform_int_distribution<size_t> idist(0, labels_vec.size()-1);

  neg_vec->clear();
  for (size_t i = 0; i < num_neg; ++i) {
    size_t idx = idist(*rnd_gen);
    neg_vec->push_back(std::make_pair(idx, 0.0f));
  }
}


void insert_pair(std::vector<std::pair<size_t, float> > &vec,
            std::pair<size_t, float> p)
{
  std::vector<std::pair<size_t, float> >::iterator it;
  for (it = vec.begin(); it < vec.end(); ++it) {
    if (it->first == p.first) {
      it->second += p.second;
      break;
    }
  }
  if (it == vec.end())
    vec.push_back(std::make_pair(p.first, p.second));
}


bool comp(std::pair<size_t, float> p1, std::pair<size_t, float> p2)
{ return (p1.first < p2.first); }


DataPartitioner::DataPartitioner() {};


DataPartitioner::~DataPartitioner() {};


void DataPartitioner::Clear() {
  K_ = 0LU;
  w_index_.clear();
};

float DataPartitioner::RunKmeans(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                                     size_t K, size_t max_iter, int seed, int verbose) {

  if (verbose > 0) {
    fprintf(stderr, "#data: %lu, K: %lu, max_iter: %lu, seed: %d\n", data_vec.size(), K, max_iter, seed);
  }

  std::mt19937 rnd_gen(seed);

  int max_fid = 1;
  std::vector<size_t> indices(data_vec.size());
  for (size_t i = 0; i < data_vec.size(); ++i) {
    indices[i] = i;
    int mfid = data_vec[i].back().first; // assume to be sorted
    if (mfid > max_fid) { max_fid = mfid; }
  }
  std::shuffle(indices.begin(), indices.end(), rnd_gen);

  // init cluster centers randomly
  HashMatrix cvec(K, max_fid + 1, mat_bit_size);
  for (size_t k = 0; k < K; ++k) {
    size_t idx = indices[k];
    for (size_t f = 0; f < data_vec[idx].size(); ++f) {
      cvec(k, data_vec[idx][f].first) = data_vec[idx][f].second;
    }
  }

  if (K == 1) {
    if (verbose > 0) { fprintf(stderr, "\rPartitioning skip!\n"); }
    return 0.0f;
  }

  std::vector<size_t> cluster_assign(data_vec.size());
  std::vector<size_t> num_assign(K);

  float prev_obj_value = -FLT_MAX;

  size_t iter = 0;
  for (iter = 0; iter < max_iter; ++iter) {
    double obj_value = 0.0;
    std::vector<size_t> k_vec(K);

    // find nearest center
    for (size_t i = 0; i < data_vec.size(); ++i) {
      obj_value += get_nearest_centers(data_vec[i], cvec, &k_vec);

      if (k_vec.size() > 1) { // tie break
        std::shuffle(k_vec.begin(), k_vec.end(), rnd_gen);
      }

      cluster_assign[i] = k_vec[0];
    }
    obj_value /= data_vec.size();

    if (verbose > 0) {
      fprintf(stderr, "\rIter: %lu, ObjectiveValue: %.6f", iter+1, obj_value);
    }

    // clear cvec
    cvec.FilledWith(0.0f);
    for (size_t k = 0; k < K; ++k) { num_assign[k] = 0; }

    // update cvec
    for (size_t i = 0; i < data_vec.size(); ++i) {
      size_t k = cluster_assign[i];
      for (size_t f = 0; f < data_vec[i].size(); ++f) {
        cvec(k, data_vec[i][f].first) += data_vec[i][f].second;
      }
      num_assign[k] += 1;
    }
    for (size_t k = 0; k < K; ++k) {
      float norm = 0.0f;
      for (size_t j = 0; j < cvec.cols(); ++j) { norm += cvec(k, j) * cvec(k, j); }
      norm = std::sqrt(norm);
      if (norm == 0.0f) { continue; }
      for (size_t j = 0; j < cvec.cols(); ++j) { cvec(k, j) /= norm; }
    }

    if (obj_value - prev_obj_value < FLT_EPSILON) { break; }
    prev_obj_value = obj_value;
  }

  // save results
  K_ = K;
  w_index_.resize(cvec.cols());
  for (size_t i = 0; i < cvec.rows(); ++i) {
    for (size_t j = 0; j < cvec.cols(); ++j) {
      int k = i;
      if (cvec(i, j) != 0.0f) { w_index_[j].push_back(std::make_pair(k, cvec(i, j))); }
    }
  }

  if (verbose > 0) {
    fprintf(stderr, "\rPartitioning done! (Iter: %lu, ObjectiveValue: %.6f)\n", iter+1, prev_obj_value);
  }

  return prev_obj_value;
}


float DataPartitioner::RunPairwise(const std::vector<std::vector<std::pair<int, float> > > &data_vec,
                                       const std::vector<std::vector<int> > &labels_vec,
                                       size_t K, size_t max_iter,
                                       size_t num_nn, int label_normalize,
                                       float eta0, float lambda, float gamma,
                                       int seed, int verbose) {
  if (verbose > 0) {
    fprintf(stderr, "#data: %lu, K: %lu, max_iter: %lu, seed: %d\n", data_vec.size(), K, max_iter, seed);
  }

  size_t cost_per_sample = 5000;

  std::mt19937 rnd_gen(seed);

  int max_fid = 1;
  std::vector<size_t> indices(data_vec.size());
  for (size_t i = 0; i < data_vec.size(); ++i) {
    indices[i] = i;
    int mfid = data_vec[i].back().first; // assume to be sorted
    if (mfid > max_fid) { max_fid = mfid; }
  }
  std::shuffle(indices.begin(), indices.end(), rnd_gen);

  // init cvec, gvec, zvec
  HashMatrix cvec(K, max_fid + 1, mat_bit_size), gvec(K, max_fid + 1, mat_bit_size), zvec(K, max_fid + 1, mat_bit_size);
  gvec.FilledWith(1.0f);
  for (size_t k = 0; k < K; ++k) {
    size_t idx = indices[k];
    for (size_t f = 0; f < data_vec[idx].size(); ++f) {
      int i = data_vec[idx][f].first;
      cvec(k, i) = data_vec[idx][f].second;
      zvec(k, i) = cvec(k, i) * std::sqrt(gvec(k, i)) / eta0;
      zvec(k, i) += (zvec(k, i) > 0.0f) ? lambda : -lambda;
    }
  }

  if (K == 1) {
    if (verbose > 0) { fprintf(stderr, "\rPartitioning skip!\n"); }
    return 0.0f;
  }

  std::vector<std::vector<std::pair<size_t, float> > > pos_vec(data_vec.size());
  std::vector<std::pair<size_t, float> > neg_vec(num_nn);

  if (max_iter > 0) {
    get_positives(labels_vec, num_nn, label_normalize, cost_per_sample, verbose, &pos_vec);
  }

  size_t pair_num = 0;
  for (size_t i = 0; i < pos_vec.size(); ++i) { pair_num += pos_vec[i].size(); }
  if (verbose > 0) {
    fprintf(stderr, "pair_num: %lu (%.2f%%)\n", pair_num, 100.0 * pair_num / num_nn / labels_vec.size());
  }

  std::vector<size_t> cluster_vec(data_vec.size());
  float prev_obj_value = 0.0f;
  size_t iter;
  for (iter = 0; iter < max_iter; ++iter) {
    size_t num_pair = 0, num_correct = 0;
    std::shuffle(indices.begin(), indices.end(), rnd_gen);
    for (size_t i = 0; i < indices.size(); ++i) {
      size_t idx = indices[i];
      sampling_negatives_uniform(labels_vec, num_nn, &rnd_gen, &neg_vec);

      num_correct += update_cvec(data_vec, idx, pos_vec[idx], neg_vec, eta0, lambda, &rnd_gen, &cvec, &gvec, &zvec, &cluster_vec);
      num_pair += pos_vec[idx].size();
    }

    float obj_value = 1.0f * num_correct / num_pair;
    if (verbose > 0) { fprintf(stderr, "\rIter: %lu, ObjectiveValue: %.6f", iter+1, obj_value); }

    prev_obj_value = obj_value;
  }

  // save results
  K_ = K;
  w_index_.resize(cvec.cols());
  size_t num_all_coef = cvec.rows() * cvec.cols();
  size_t nnz = 0;
  for (size_t i = 0; i < cvec.rows(); ++i) {
    for (size_t j = 0; j < cvec.cols(); ++j) {
      int k = i;
      if (std::fabs(cvec(i, j)) > DBL_EPSILON) {
        w_index_[j].push_back(std::make_pair(k, cvec(i, j)));
        ++nnz;
      }
    }
  }

  if (verbose > 0) {
    fprintf(stderr, "\rPartitioning done! (Iter: %lu, ObjectiveValue: %.6f)\n", iter+1, prev_obj_value);
    fprintf(stderr, "nnz: %lu (in %lu, %.4f%%)\n", nnz, num_all_coef, 100.0f * nnz / num_all_coef);
  }

  return prev_obj_value;
}


float  DataPartitioner::RunNeighbourExpansionEP(const std::vector<std::vector<int> > &labels_vec,
                                             std::vector<std::set<size_t> > &cluster_assign,
                                             size_t K, size_t num_nn, int label_normalize,
                                             float replication_factor, int seed, int verbose) 
{
  if (verbose > 0) {
    fprintf(stderr, "#data: %lu, K: %lu, seed: %d\n", labels_vec.size(), K, seed);
  }

  srand(seed);
  size_t cost_per_sample = 5000;

  if (K == 1) {
    if (verbose > 0) { fprintf(stderr, "\rPartitioning skip!\n"); }
    cluster_assign.resize(1);
    for (auto i = 0; i < labels_vec.size(); i++)
      cluster_assign[0].insert(i);
    return 0.0;
  }

  std::vector<std::vector<std::pair<size_t, float> > > pos_vec(labels_vec.size());

  get_positives(labels_vec, num_nn, label_normalize, cost_per_sample, verbose, &pos_vec);

#define SYM 1
#if SYM
  // create a symmetric nn graph
  std::vector<std::set<size_t> > nn_graph(labels_vec.size());
  for (auto i = 0; i < pos_vec.size(); ++i) {
    std::unordered_set<size_t> vset;
    for (auto&& p: pos_vec[i]) {
      size_t cur = p.first;
      if (std::find(vset.begin(), vset.end(), cur) != vset.end())
        continue;
      vset.insert(cur);
      if (std::find(nn_graph[i].begin(), nn_graph[i].end(), cur) == nn_graph[i].end())
        nn_graph[i].insert(cur);
      if (std::find(nn_graph[cur].begin(), nn_graph[cur].end(), i) == nn_graph[cur].end())
        nn_graph[cur].insert(i);
    }
  }
#else
  // create asymmetric nn graph
  std::vector<std::set<size_t> > nn_graph(labels_vec.size());
  for (auto i = 0; i < pos_vec.size(); ++i) {
    for (auto&& p: pos_vec[i])
      nn_graph[i].insert(p.first);
  }
#endif //#if


  pos_vec.clear();

  size_t pair_num = 0;
  for (size_t i = 0; i < nn_graph.size(); ++i) { 
    pair_num += nn_graph[i].size();
  }

  if (verbose > 0) {
    fprintf(stderr, "# of nn graph edges: %lu (%.2f%%)\n", pair_num, 100.0 * pair_num / num_nn / labels_vec.size());
  }

  // partition dataset by edge partitioning
  cluster_assign.clear();
  cluster_assign.resize(K);
  std::set<size_t> left_vertices;
  for (size_t i = 0; i < labels_vec.size(); ++i)
    left_vertices.insert(i);

#if SYM
  float delta_max = (replication_factor * pair_num) / (K * 2);
#else
  float delta_max = (replication_factor * pair_num) / K;
#endif
  float delta_min = (float)labels_vec.size() / K;
  for (auto k = 0; k < K; k++) {
    get_edge_set(nn_graph, cluster_assign[k], left_vertices, delta_min, delta_max);
    if (verbose > 0) {
      fprintf(stderr, "cluster %d: sample size %lu\n", k, cluster_assign[k].size());
    }
  }

  size_t pair_left = 0;
  for (auto&& vec: nn_graph) {
    pair_left += vec.size();
  }

  float rep_factor = 0;
  for (auto k = 0; k < K; k++)
    rep_factor += cluster_assign[k].size();
  rep_factor /= labels_vec.size();

  if (verbose > 0) {
    fprintf(stderr, "# of edges lost: %lu (%.2f%%)\n", pair_left, (100.0 * pair_left) / pair_num);
  }
  
  nn_graph.clear();

  return rep_factor;
}


size_t DataPartitioner::GetNearestCluster(const std::vector<std::pair<int, float> > &datum) const {
  std::vector<size_t> centers;
  GetNearestClusters(datum, &centers);
  return centers[0];
}


float DataPartitioner::GetNearestClusters(const std::vector<std::pair<int, float> > &datum,
                                              std::vector<size_t> *centers) const {
  if (centers == NULL) { return 0.0f; }

  std::vector<float> ip_vec(K_, 0.0f);

  for (size_t f = 0; f < datum.size(); ++f) {
    size_t idx = datum[f].first;
    if (idx >= w_index_.size()) { continue; }

    for (size_t i = 0; i < w_index_[idx].size(); ++i) {
      size_t k = w_index_[idx][i].first;
      ip_vec[k] += datum[f].second * w_index_[idx][i].second;
    }
  }

  float max_ip = -FLT_MAX;
  for (size_t k = 0; k < K_; ++k) {
    float ip = ip_vec[k];
    if (ip > max_ip) {
      centers->clear();
      centers->push_back(k);
      max_ip = ip;
    } else if (std::fabs(max_ip - ip) < FLT_EPSILON) {
      centers->push_back(k);
    }
  }

  return max_ip;
}


int DataPartitioner::NormalizeData(std::vector<std::vector<std::pair<int, float> > > *data_vec) const {
  for (size_t i = 0; i < data_vec->size(); ++i) {
    float norm = 0.0f;
    for (size_t f = 0; f < (*data_vec)[i].size(); ++f) {
      float v = (*data_vec)[i][f].second;
      norm += v * v;
    }
    if (norm == 0.0f) { continue; }
    norm = std::sqrt(norm);

    for (size_t f = 0; f < (*data_vec)[i].size(); ++f) {
      (*data_vec)[i][f].second /= norm;
    }
  }

  return 1;
}

void DataPartitioner::CopiedFrom(const DataPartitioner &that) {
  K_ = that.K_;
  w_index_.resize(that.w_index_.size());
  for (size_t i = 0; i < w_index_.size(); ++i) {
    w_index_[i].resize(that.w_index_[i].size());
    for (size_t j = 0; j < that.w_index_[i].size(); ++j) {
      w_index_[i][j] = that.w_index_[i][j];
    }
  }
}

int DataPartitioner::WriteToStream(FILE *stream) const {
  yj::xmlc::Utils::WriteNumToStream(K_, stream);
  yj::xmlc::Utils::WriteNumToStream(w_index_.size(), stream);
  for (size_t i = 0; i < w_index_.size(); ++i) {
    yj::xmlc::Utils::WriteNumToStream(w_index_[i].size(), stream);
    for (size_t j = 0; j < w_index_[i].size(); ++j) {
      yj::xmlc::Utils::WriteNumToStream(w_index_[i][j].first, stream);
      yj::xmlc::Utils::WriteNumToStream(w_index_[i][j].second, stream);
    }
  }

  return 1;
}


int DataPartitioner::ReadFromStream(FILE *stream) {
  size_t vec_size;
  w_index_.clear();

  yj::xmlc::Utils::ReadNumFromStream(stream, &K_);
  yj::xmlc::Utils::ReadNumFromStream(stream, &vec_size);
  w_index_.resize(vec_size);

  for (size_t i = 0; i < w_index_.size(); ++i) {
    yj::xmlc::Utils::ReadNumFromStream(stream, &vec_size);
    w_index_[i].resize(vec_size);
    for (size_t j = 0; j < vec_size; ++j) {
      yj::xmlc::Utils::ReadNumFromStream(stream, &(w_index_[i][j].first));
      yj::xmlc::Utils::ReadNumFromStream(stream, &(w_index_[i][j].second));
    }
  }

  return 1;
}

