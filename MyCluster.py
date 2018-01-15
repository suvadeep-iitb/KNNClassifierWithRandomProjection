import numpy as np
from data_partitioner import DataPartitioner as DP
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import LinearSVC
from datetime import datetime
import minmax_kmeans as mmkmeans

class NearestNeighbour:
  def __init__(self,
               n_clusters,
               max_iter,
               seed,
               verbose,
               num_nn,
               label_normalize,
               eta0,
               lamb):
    self.n_clusters_ = n_clusters
    self.max_iter_ = max_iter
    self.seed_ = seed
    self.verbose_ = verbose
    self.num_nn_ = num_nn
    self.label_normalize_ = label_normalize
    self.eta0_ = eta0
    self.lambda_ = lamb
    self.dataPartitioner_ = DP()


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])

    X = csr_matrix(X)
    Y = csr_matrix(Y)

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    Y_indices = Y.indices
    Y_indptr = Y.indptr

    self.dataPartitioner_.RunPairwise(X_indices,
                                      X_data,
                                      X_indptr,
                                      Y_indices,
                                      Y_indptr,
                                      self.n_clusters_,
                                      self.max_iter_,
                                      self.num_nn_,
                                      self.label_normalize_,
                                      self.eta0_,
                                      self.lambda_,
                                      0, # Currently the parameter 'gamma is not being used'
                                      self.seed_,
                                      self.verbose_)
    self.labels_ = np.zeros((X.shape[0]), dtype=np.int32)
    self.dataPartitioner_.GetNearestClusters(X_indices, X_data, X_indptr, self.labels_)


  def predict(self, X):
    X = csr_matrix(X)

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    labels = np.zeros((X.shape[0]), dtype=np.int32)
    self.dataPartitioner_.GetNearestClusters(X_indices, X_data, X_indptr, labels)

    return labels


class MinMaxKMeans:
  def __init__(self,
               n_clusters,
               max_iter = 10,
               n_init = 1,
               min_size = 0,
               max_size = None,
               seed = 0,
               verbose = 1,
               n_jobs = 1):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_ = max_iter
    self.min_size_ = min_size
    self.max_size_ = max_size
    self.seed_ = seed
    self.verbose_ = verbose
    self.n_jobs_ = n_jobs


  def fit(self, X):
    X = list(X)
    for i in range(self.n_init_):
      clusters, centers = mmkmeans.minsize_kmeans(X, self.n_clusters_, self.max_iter_, self.min_size_, self.max_size_)
      if clusters:
        quality = mmkmean.compute_quality(X, clusters)
        if not best or (quality < best):
          best = quality
          best_clusters = clusters
          best_centers = centers
    self.labels_ = np.array(best_clusters)
    self.centers_ = np.array(best_centers).reshape((self.n_clusters_, X.shape[1]))


  def predict(self, X):
    dist = cdist(X, self.centers_.T)
    return np.argmin(dist, axis = 1).reshape(-1)
    


class LabelKmeans:
  def __init__(self,
               n_clusters,
               n_init,
               max_iter,
               C,
               seed,
               verbose,
               label_normalize,
               alpha,
               n_jobs):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_ = max_iter
    self.C_ = C
    self.seed_ = seed
    self.verbose_ = verbose
    self.label_normalize_ = label_normalize
    self.alpha_ = alpha
    self.n_jobs_ = n_jobs

  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])

    self.base_clusters_ = KMeans(n_clusters = self.n_clusters_,
                                       n_init = self.n_init_,
                                       max_iter = self.max_iter_,
                                       random_state = self.seed_,
                                       #min_size = X.shape[0]/float(2*self.n_clusters_),
                                       verbose = self.verbose_,
                                       n_jobs = self.n_jobs_
                                       )
    self.clf_ = LinearSVC(dual = False, C=self.C_)

    label_freq = np.array(np.sum(Y, axis = 0)).reshape(-1)
    freq_cutoff = self.alpha_ * Y.shape[0]
    Y = Y[:, label_freq < freq_cutoff]
    print(str(np.sum(label_freq > freq_cutoff))+' labels have been removed during clustering')

    X = csr_matrix(X)
    Y = csr_matrix(Y)
    if (self.label_normalize_):
      Y = normalize(Y, norm = 'l2', axis = 0)

    self.base_clusters_.fit(Y.T)
    self.cluster_assignments_ = []
    for cid in range(self.n_clusters_):
      sel_labels = (self.base_clusters_.labels_ == cid)
      cl_ass = (np.sum(Y[:, sel_labels], axis=1) > 0.0)
      self.cluster_assignments_.append(cl_ass)
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(np.sum(cl_ass))+' # of labels '+str(np.sum(sel_labels)))

    self.centers_ = np.zeros((0, X.shape[1]+1), dtype=np.float)
    print(str(datetime.now())+' : Learning predictor for each cluster')
    for cid, cl_ass in enumerate(self.cluster_assignments_):
      Xtr, Xte, Ytr, Yte = train_test_split(X, cl_ass, test_size = 0.1, random_state = self.seed_)
      self.clf_.fit(Xtr, np.array(Ytr).reshape(-1))
      self.centers_ = np.vstack([self.centers_, np.hstack([self.clf_.coef_, self.clf_.intercept_.reshape((1, 1))])])
      print(str(datetime.now())+' : Cluster '+str(cid)+' train accuracy '+str(self.clf_.score(Xtr, Ytr))+' test accuracy '+str(self.clf_.score(Xte, Yte)))

    print(str(datetime.now())+' : Computing label assignment for each example')
    self.labels_ = self.predict(X)



  def predict(self, X):
    X = csr_matrix(X)
    scores = X * self.centers_[:, :-1].T + self.centers_[:, -1].T
    return np.argmax(scores, axis=1).reshape(-1)


def CompressDimension(X):
  colSum = np.sum(np.abs(X), axis=0)
  _, idMapping = np.where(colSum > 0.0)
  X = X[:, idMapping]
  return X, idMapping
