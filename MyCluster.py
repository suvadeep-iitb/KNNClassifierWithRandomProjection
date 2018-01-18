import numpy as np
from data_partitioner import DataPartitioner as DP
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, vstack, hstack
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import LinearSVC
from MulticlassPredictor import MulticlassPredictor
from datetime import datetime
import minmax_kmeans as mmkmeans
#import minmax_kmeans as mmkmeans
import pickle


'''
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


  def load_model(self, filename):
    centers = pickle.load(open(filename, 'rb'))
    self.dataPartitioner_.clear()
    self.dataPartitioner_ = DP(centers)


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])
    self.n_features_ = X.shape[1]

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


  def get_clusters(self):
    centers = np.zeros((self.n_clusters_, self.n_features_), dtype=np.float)
    self.dataPartitioner_.GetCenters(centers)
    return centers



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
'''

    
class LabelRand:
  def __init__(self,
               n_clusters,
               n_init,
               max_iter,
               C,
               seed,
               verbose,
               alpha,
               n_jobs):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_ = max_iter
    self.C_ = C
    self.seed_ = seed
    self.verbose_ = verbose
    self.alpha_ = alpha
    self.n_jobs_ = n_jobs


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])

    params = {'lamb': self.C_, 'itr': self.max_iter_}
    self.clf_ = MulticlassPredictor(params)

    label_freq = np.array(np.sum(Y, axis = 0)).reshape(-1)
    freq_cutoff = self.alpha_ * Y.shape[0]
    Y = Y[:, label_freq < freq_cutoff]
    print(str(np.sum(label_freq > freq_cutoff))+' labels have been removed during clustering')

    assignments = [l%self.n_clusters_ for l in range(Y.shape[1])]
    assignments = np.random.permutation(assignments)
    self.cluster_assignments_ = csc_matrix((X.shape[0], 0), dtype=np.float)
    for cid in range(self.n_clusters_):
      sel_labels = (assignments == cid)
      cl_ass = (np.sum(Y[:, sel_labels], axis=1) > 0.0).reshape((X.shape[0], 1))
      self.cluster_assignments_ = hstack([self.cluster_assignments_, csc_matrix(cl_ass)])
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(np.sum(cl_ass))+' # of labels '+str(np.sum(sel_labels)))
    self.cluster_assignments_ = csr_matrix(self.cluster_assignments_)

    Xtr, Xte, Ytr, Yte = train_test_split(X, self.cluster_assignments_, test_size = 0.1, random_state = self.seed_)
    self.clf_.Train(Xtr, Ytr, numThreads = self.n_jobs_)
    labels, _ = self.clf_.Predict(Xtr, numThreads = self.n_jobs_)
    labels = np.array(labels[:, 0].todense()).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in train set '+str(np.sum([Ytr[i, labels[i]] for i in range(Ytr.shape[0])])/float(Ytr.shape[0])))
    labels, _ = self.clf_.Predict(Xte, numThreads = self.n_jobs_)
    labels = np.array(labels[:, 0].todense()).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in valid set '+str(np.sum([Yte[i, labels[i]] for i in range(Yte.shape[0])])/float(Yte.shape[0])))
  
    pickle.dump(self.clf_.GetParamMatrix(), open('WeightMatrix.pkl', 'wb'))
    print(str(datetime.now())+' : Computing label assignment for each example')
    labels, _ = self.clf_.Predict(X, numThreads = self.n_jobs_)
    self.labels_ = np.array(labels[:, 0].todense()).reshape(-1)


  def predict(self, X):
    labels, _ = self.clf_.Predict(X, numThreads = self.n_jobs_)
    return np.array(labels[:, 0].todense()).reshape(-1)



class LabelRand:
  def __init__(self,
               n_clusters,
               n_init,
               max_iter_kmeans,
               C,
               max_iter_svc,
               seed,
               verbose,
               log_file,
               label_normalize,
               alpha,
               n_jobs):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_kmeans_ = max_iter_kmeans
    self.C_ = C
    self.max_iter_svc_ = max_iter_svc
    self.seed_ = seed
    self.verbose_ = verbose
    self.log_file_ = log_file
    self.label_normalize_ = label_normalize
    self.alpha_ = alpha
    self.n_jobs_ = n_jobs
    self.log_ = ''


  def cluster_labels(self, X, Y):
    self.base_clusters_ = KMeans(n_clusters = self.n_clusters_,
                                 n_init = self.n_init_,
                                 max_iter = self.max_iter_kmeans_,
                                 random_state = self.seed_,
                                 verbose = self.verbose_,
                                 n_jobs = self.n_jobs_)
    label_freq = np.array(np.sum(Y, axis = 0)).reshape(-1)
    freq_cutoff = self.alpha_ * Y.shape[0]
    Y = Y[:, label_freq < freq_cutoff]
    print(str(np.sum(label_freq > freq_cutoff))+' labels have been removed during clustering')
    self.log_ += str(np.sum(label_freq > freq_cutoff))+' labels have been removed during clustering\n'
    Y = csr_matrix(Y)
    if (self.label_normalize_):
      Y = normalize(Y, norm = 'l2', axis = 0)

    labels = np.array([i % self.n_clusters_ for i in range(Y.shape[1])])
    labels = np.random.permutation(labels)
    self.cluster_assignments_ = []
    for cid in range(self.n_clusters_):
      sel_labels = (labels == cid)
      cl_ass = csr_matrix(np.sum(Y[:, sel_labels], axis=1) > 0.0).reshape(-1, 1)
      self.cluster_assignments_.append(cl_ass)
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(np.sum(cl_ass))+' # of labels '+str(np.sum(sel_labels)))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(np.sum(cl_ass))+' # of labels '+str(np.sum(sel_labels))+'\n'
    self.cluster_assignments_ = hstack(self.cluster_assignments_)


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])

    self.cluster_labels(X, Y)
 
    params = {'lamb': self.C_, 'itr': self.max_iter_svc_}
    self.clf_ = MulticlassPredictor(params)

    self.centers_ = np.zeros((0, X.shape[1]+1), dtype=np.float)
    print(str(datetime.now())+' : Learning predictor for each cluster')
    self.log_ += str(datetime.now())+' : Learning predictor for each cluster\n'

    Xtr, Xte, Ytr, Yte = train_test_split(X, self.cluster_assignments_, test_size = 0.1, random_state = self.seed_)
    self.clf_.Train(Xtr, Ytr, numThreads = self.n_jobs_)

    labels, _ = self.clf_.Predict(Xtr, numThreads = self.n_jobs_)
    labels = np.array(labels[:, 0].todense()).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in train set '+str(np.sum([Ytr[i, labels[i]] for i in range(Ytr.shape[0])])/float(Ytr.shape[0])))
    self.log_ += str(datetime.now())+' : Cluster selection accuracy in train set '+str(np.sum([Ytr[i, labels[i]] for i in range(Ytr.shape[0])])/float(Ytr.shape[0]))+'\n'

    labels, _ = self.clf_.Predict(Xte, numThreads = self.n_jobs_)
    labels = np.array(labels[:, 0].todense()).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in valid set '+str(np.sum([Yte[i, labels[i]] for i in range(Yte.shape[0])])/float(Yte.shape[0])))
    self.log_ += str(datetime.now())+' : Cluster selection accuracy in valid set '+str(np.sum([Yte[i, labels[i]] for i in range(Yte.shape[0])])/float(Yte.shape[0]))+'\n'

    del self.cluster_assignments_, Xtr, Xte, Ytr, Yte
    self.centers_ = self.clf_.W

    print(str(datetime.now())+' : Computing label assignment for each example')
    self.log_ += str(datetime.now())+' : Computing label assignment for each example\n'
    labels, _ = self.clf_.Predict(X, numThreads = self.n_jobs_)
    self.labels_ = np.array(labels[:, 0].todense()).reshape(-1)

    if (self.log_file_):
      pickle.dump(self.log_, open(self.log_file_, 'wb'))


  def predict(self, X):
    labels, _ = self.clf_.Predict(X, numThreads = self.n_jobs_)
    return np.array(labels[:, 0].todense()).reshape(-1)



class LabelNeighbourExpensionEP(LabelRand):
  def __init__(self,
               n_clusters,
               n_init,
               num_nn,
               rep_factor,
               label_normalize,
               C,
               max_iter_svc,
               seed,
               verbose,
               log_file,
               n_jobs):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.num_nn_ = num_nn
    self.rep_factor_ = rep_factor
    self.C_ = C
    self.max_iter_svc_ = max_iter_svc
    self.seed_ = seed
    self.verbose_ = verbose
    self.log_file_ = log_file
    self.label_normalize_ = label_normalize
    self.n_jobs_ = n_jobs
    self.log_ = ''


  def cluster_labels(self, X, Y):
    Y = csr_matrix(Y)
    indices = Y.indices
    indptr = Y.indptr
    print(str(datetime.now())+' : Starting RunNeighbourExpansionEP partitioning on label matrix')
    self.log_ += str(datetime.now())+' : Starting RunNeighbourExpansionEP partitioning on label matrix\n'
    self.cluster_assignments_ = np.zeros((self.n_clusters_, Y.shape[0]), dtype=np.float)
    dp =DP()
    obj_value = dp.RunNeighbourExpansionEP(indices, indptr, 
                             self.cluster_assignments_,
                             self.n_clusters_, self.num_nn_,
                             self.label_normalize_, self.rep_factor_,
                             self.seed_, self.verbose_)

    self.cluster_assignments_ = self.cluster_assignments_.T
    for cid in range(self.n_clusters_):
      cl_ass = np.array(self.cluster_assignments_[:, cid]).reshape(-1)
      sel_labels = (np.sum(Y[cl_ass > 0, :], axis=0) > 0)
      print(str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(np.sum(cl_ass))+', # of labels '+str(int(np.sum(sel_labels))))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(np.sum(cl_ass))+', # of labels '+str(int(np.sum(sel_labels)))+'\n'
    self.cluster_assignments_ = csr_matrix(self.cluster_assignments_)
    pickle.dump(self.cluster_assignments_, open('clus_ass.pkl', 'wb'));



def CompressDimension(X):
  colSum = np.sum(np.abs(X), axis=0)
  _, idMapping = np.where(colSum > 0.0)
  X = X[:, idMapping]
  return X, idMapping
