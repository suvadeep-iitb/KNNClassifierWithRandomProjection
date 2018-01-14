import numpy as np
from data_partitioner import DataPartitioner as DP
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from datetime import datetime

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



class LabelKmeans:
  def __init__(self,
               n_clusters,
               n_init,
               max_iter,
               seed,
               verbose,
               label_normalize,
               lamb,
               n_jobs):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_ = max_iter
    self.seed_ = seed
    self.verbose_ = verbose
    self.label_normalize_ = label_normalize
    self.lamb_ = lamb
    self.n_jobs_ = n_jobs
    self.base_clusters_ = KMeans(n_clusters = n_clusters,
                                n_init = n_init,
                                max_iter = max_iter,
                                random_state = seed,
                                verbose = verbose,
                                n_jobs = n_jobs)


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])

    X = csr_matrix(X)
    Y = csr_matrix(Y)
    if (self.label_normalize_):
      Y = normalize(Y, norm = 'l2', axis = 1)

    self.base_clusters_.fit(Y.T)
    self.cluster_assignments_ = []
    for cid in range(self.n_clusters_):
      sel_labels = (self.base_clusters_.labels_ == cid)
      cl_ass = (np.sum(Y[:, sel_labels], axis=1) > 0.0)
      self.cluster_assignments_.append(cl_ass)
      print(str(datatime.now())+' : Cluster '+str(cid)+' # of examples '+str(np.sum(cl_ass))+' # of labels '+str(np.sum(self.base_clusters_.assignments_==cid)))

    clf = LinearSVC(dual = False, C=1.0, intercept = False)
    self.centers_ = np.zeros((0, X.shape[1]), dtype=np.float)
    print(str(datetime.now())+' : Learning predictor for each cluster')
    for cid, cl_ass in enumerate(self.cluster_assignments_):
      clf.fit(X, cl_ass)
      self.centers_ = vstack([self.centers_, clf.coef_.reshape(1, -1)])
      print(str(datetime.now())+' : Cluster '+str(cid)+' mean accuracy '+str(clf.score(X, cl_ass)))

    print(str(datetime.now())+' : Computing label assignment for each example')
    self.labels_ = self.predict(X)



  def predict(self, X):
    X = csr_matrix(X)
    scores = X * self.centers_.T
    return np.argmax(scores, axis=1).reshape(-1)


def CompressDimension(X):
  colSum = np.sum(np.abs(X), axis=0)
  _, idMapping = np.where(colSum > 0.0)
  X = X[:, idMapping]
  return X, idMapping
