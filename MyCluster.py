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
#import minmax_kmeans as mmkmeans
import pickle


class NearestNeighbour:
  def __init__(self,
               n_clusters,
               n_init = 1,
               max_iter = 10,
               num_nn = 10,
               label_normalize = 1,
               seed = 1,
               log_file = '',
               verbose = 1,
               eta0 = 0.1,
               lamb = 4.0):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.max_iter_ = max_iter
    self.seed_ = seed
    self.log_file_ = log_file
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


  def update_seed(self, seed):
    self.seed_ = seed


  def update_log_file(self, log_file):
    self.log_file_ = log_file


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
    print(str(datetime.now())+' : Final cluster assignments:')
    self.Y_ = []
    self.label_mapping_ = np.array(range(self.n_clusters_))
    for cid in range(self.n_clusters_):
      cl_ass = np.array(self.labels_ == cid).reshape(-1)
      if (np.sum(cl_ass) == 0):
        for lid in range(cid+1, self.n_clusters_):
          self.label_mapping_[lid] -= 1
      sel_labels = (np.sum(Y[cl_ass, :], axis = 0) > 0)
      self.Y_.append(sel_labels)
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(int(np.sum(cl_ass)))+' # of labels '+str(np.sum(sel_labels)))
    for i in range(self.labels_.shape[0]):
      self.labels_[i] = self.label_mapping_[self.labels_[i]]



  def predict(self, X):
    X = csr_matrix(X)

    X_data = X.data
    X_indices = X.indices
    X_indptr = X.indptr

    labels = np.zeros((X.shape[0]), dtype=np.int32)
    self.dataPartitioner_.GetNearestClusters(X_indices, X_data, X_indptr, labels)
    for i in range(labels.shape[0]):
      labels[i] = self.label_mapping_[labels[i]]

    return labels


  def get_clusters(self):
    raise NotImplemented


'''
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
               n_init = 1,
               C = 1,
               max_iter_svc = 50,
               seed = 1,
               verbose = 1,
               log_file = '',
               center_file = '',
               clus_ass_file = '',
               n_jobs = 1):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.C_ = C
    self.max_iter_svc_ = max_iter_svc
    self.seed_ = seed
    self.verbose_ = verbose
    self.log_file_ = log_file
    self.n_jobs_ = n_jobs
    self.center_file_ = center_file
    self.clus_ass_file_ = clus_ass_file
    self.log_ = ''


  def update_seed(self, seed):
    self.seed_ = seed


  def update_log_file(self, log_file):
    self.log_file_ = log_file


  def cluster_labels(self, X, Y):
    Y = csr_matrix(Y)

    np.random.seed(self.seed_)
    labels = np.array([i % self.n_clusters_ for i in range(Y.shape[1])])
    labels = np.random.permutation(labels)
    self.cluster_assignments_ = []
    for cid in range(self.n_clusters_):
      sel_labels = (labels == cid)
      cl_ass = csr_matrix((np.sum(Y[:, sel_labels], axis=1) > 0).reshape(-1, 1), dtype = np.int32)
      self.cluster_assignments_.append(cl_ass)
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(int(np.sum(cl_ass)))+' # of labels '+str(np.sum(sel_labels)))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(int(np.sum(cl_ass)))+' # of labels '+str(np.sum(sel_labels))+'\n'
    self.cluster_assignments_ = csr_matrix(hstack(self.cluster_assignments_))
    clus_sizes = np.array(self.cluster_assignments_.sum(0)).reshape(-1)
    self.cluster_assignments_ = self.cluster_assignments_[:, clus_sizes>0]
    self.n_clusters_ = self.cluster_assignments_.shape[1]


  def fit(self, X, Y):
    assert(X.shape[0] == Y.shape[0])
    params = {'lamb': self.C_, 'itr': self.max_iter_svc_}
    self.clf_ = MulticlassPredictor(params)

    if self.center_file_:
      print(str(datetime.now())+' : Loading cluster centers from file: '+self.center_file_)
      self.log_ += str(datetime.now())+' : Loading cluster centers from file: '+self.center_file_+'\n'
      self.centers_ = pickle.load(open(self.center_file_, 'rb'))
      assert(self.centers_.shape[0] == X.shape[1]+1)
      assert(self.centers_.shape[1] == self.n_clusters_)
      self.clf_.LoadModel(self.centers_)
      print(str(datetime.now())+' : Computing label assignment for each example')
      self.log_ += str(datetime.now())+' : Computing label assignment for each example\n'
      predY = csr_matrix(self.clf_.Predict(X, numThreads = self.n_jobs_))
      self.labels_ = np.array(predY.argmax(1)).reshape(-1)

      if (self.log_file_):
        pickle.dump(self.log_, open(self.log_file_, 'wb'))
      return

    self.cluster_labels(X, Y)

    # remove unassigned samples from the train dataset
    print(str(datetime.now())+' : Removing unassigned samples from the train dataset')
    self.log_ += str(datetime.now())+' : Removing unassigned samples from the train dataset\n'
    assigned_samples = (np.array(np.sum(self.cluster_assignments_, axis = 1)).reshape(-1) > 0)
    print(str(datetime.now())+' : # of Unassigned samples '+str(assigned_samples.shape[0]-np.sum(assigned_samples)))
 
    print(str(datetime.now())+' : Learning predictor for each cluster')
    self.log_ += str(datetime.now())+' : Learning predictor for each cluster\n'

    Xtr, Xte, Ytr, Yte = train_test_split(X[assigned_samples, :], self.cluster_assignments_[assigned_samples, :], test_size = 0.1, random_state = self.seed_)
    self.clf_.Train(Xtr, Ytr, numThreads = self.n_jobs_)

    predYtr = csr_matrix(self.clf_.Predict(Xtr, numThreads = self.n_jobs_))
    labels = np.array(predYtr.argmax(1)).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in train set '+str(np.sum([Ytr[i, labels[i]] for i in range(Ytr.shape[0])])/float(Ytr.shape[0])))
    self.log_ += str(datetime.now())+' : Cluster selection accuracy in train set '+str(np.sum([Ytr[i, labels[i]] for i in range(Ytr.shape[0])])/float(Ytr.shape[0]))+'\n'

    predYte = csr_matrix(self.clf_.Predict(Xte, numThreads = self.n_jobs_))
    labels = np.array(predYte.argmax(1)).reshape(-1)
    print(str(datetime.now())+' : Cluster selection accuracy in valid set '+str(np.sum([Yte[i, labels[i]] for i in range(Yte.shape[0])])/float(Yte.shape[0])))
    self.log_ += str(datetime.now())+' : Cluster selection accuracy in valid set '+str(np.sum([Yte[i, labels[i]] for i in range(Yte.shape[0])])/float(Yte.shape[0]))+'\n'

    print(str(datetime.now())+' : Computing label assignment for each example')
    self.log_ += str(datetime.now())+' : Computing label assignment for each example\n'
    predY = csr_matrix(self.clf_.Predict(X, numThreads = self.n_jobs_))
    self.labels_ = np.array(predY.argmax(1)).reshape(-1)

    self.centers_ = self.clf_.W
    del Xtr, Xte, Ytr, Yte, predYte, predYtr, labels

    print(str(datetime.now())+' : Final cluster assignments:')
    self.log_ += str(datetime.now())+' : Final cluster assignments:\n'
    self.Y_ = []
    for cid in range(self.n_clusters_):
      cl_ass = np.array(self.labels_ == cid).reshape(-1)
      sel_labels = (np.sum(Y[cl_ass, :], axis = 0) > 0)
      self.Y_.append(sel_labels)
      print(str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(int(np.sum(cl_ass)))+' # of labels '+str(np.sum(sel_labels)))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+' # of examples '+str(int(np.sum(cl_ass)))+' # of labels '+str(np.sum(sel_labels))+'\n'

    if (self.log_file_):
      pickle.dump(self.log_, open(self.log_file_, 'wb'))


  def predict(self, X):
    predY = csr_matrix(self.clf_.Predict(X, numThreads = self.n_jobs_))
    return np.array(predY.argmax(1)).reshape(-1)



class LabelNeighbourExpensionEP(LabelRand):
  def __init__(self,
               n_clusters,
               n_init = 1,
               num_nn = 10,
               rep_factor = 1.1,
               label_normalize = 1,
               C = 1,
               max_iter_svc = 50,
               seed = 1,
               verbose = 1,
               log_file = '',
               res_file = '',
               center_file = '',
               clus_ass_file = '',
               n_jobs = 1):
    self.n_clusters_ = n_clusters
    self.n_init_ = n_init
    self.num_nn_ = num_nn
    self.rep_factor_ = rep_factor
    self.C_ = C
    self.max_iter_svc_ = max_iter_svc
    self.seed_ = seed
    self.verbose_ = verbose
    self.log_file_ = log_file
    self.res_file_ = res_file
    self.center_file_ = center_file
    self.clus_ass_file_ = clus_ass_file
    self.label_normalize_ = label_normalize
    self.n_jobs_ = n_jobs
    self.log_ = ''


  def cluster_labels(self, X, Y):
    if (self.clus_ass_file_):
      self.cluster_assignments_ = pickle.load(open(self.clus_ass_file_, 'rb'))
      return

    X = csr_matrix(X)
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    Y = csr_matrix(Y)
    Y_indices = Y.indices
    Y_indptr = Y.indptr

    print(str(datetime.now())+' : Starting RunNeighbourExpansionEP partitioning on label matrix')
    self.log_ += str(datetime.now())+' : Starting RunNeighbourExpansionEP partitioning on label matrix\n'
    self.cluster_assignments_ = np.zeros((self.n_clusters_, Y.shape[0]), dtype=np.float)
    dp =DP()
    obj_value = dp.RunNeighbourExpansionEP(X_indices, X_data, X_indptr,
                                           Y_indices, Y_indptr, 
                                           self.cluster_assignments_,
                                           self.n_clusters_, self.num_nn_,
                                           self.label_normalize_, self.rep_factor_,
                                           self.seed_, self.verbose_)

    self.cluster_assignments_ = self.cluster_assignments_.T
    for cid in range(self.n_clusters_):
      cl_ass = np.array(self.cluster_assignments_[:, cid]).reshape(-1)
      sel_labels = (np.sum(Y[cl_ass > 0, :], axis=0) > 0)
      print(str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(int(np.sum(cl_ass)))+', # of labels '+str(int(np.sum(sel_labels))))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(int(np.sum(cl_ass)))+', # of labels '+str(int(np.sum(sel_labels)))+'\n'
    self.cluster_assignments_ = csr_matrix(self.cluster_assignments_)
    clus_sizes = np.array(self.cluster_assignments_.sum(0)).reshape(-1)
    self.cluster_assignments_ = self.cluster_assignments_[:, clus_sizes>0]
    self.n_clusters_ = self.cluster_assignments_.shape[1]

    if (self.res_file_):
      pickle.dump(self.cluster_assignments_, open(self.res_file_, 'wb'));



class LabelNeighbourExpensionVP(LabelNeighbourExpensionEP):

  def cluster_labels(self, X, Y):
    if (self.clus_ass_file_):
      self.cluster_assignments_ = pickle.load(open(self.clus_ass_file_, 'rb'))
      return

    X = csr_matrix(X)
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    Y = csr_matrix(Y)
    Y_indices = Y.indices
    Y_indptr = Y.indptr
    print(str(datetime.now())+' : Starting RunNeighbourExpansionVP partitioning on label matrix')
    self.log_ += str(datetime.now())+' : Starting RunNeighbourExpansionVP partitioning on label matrix\n'
    clus_assign = np.zeros((Y.shape[0]), dtype=np.int32)
    dp =DP()
    obj_value = dp.RunNeighbourExpansionVP(X_indices, X_data, X_indptr,
                                           Y_indices, Y_indptr, clus_assign,
                                           self.n_clusters_, self.num_nn_,
                                           self.label_normalize_, self.rep_factor_,
                                           self.seed_, self.verbose_)

    self.cluster_assignments_ = [];
    for cid in range(self.n_clusters_):
      cl_ass = (clus_assign == cid);
      sel_labels = (np.sum(Y[cl_ass > 0, :], axis=0) > 0)
      self.cluster_assignments_.append(csr_matrix(cl_ass.reshape((-1, 1))))
      print(str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(int(np.sum(cl_ass)))+', # of labels '+str(int(np.sum(sel_labels))))
      self.log_ += str(datetime.now())+' : Cluster '+str(cid)+', # of examples '+str(int(np.sum(cl_ass)))+', # of labels '+str(int(np.sum(sel_labels)))+'\n'

    self.cluster_assignments_ = csr_matrix(hstack(self.cluster_assignments_))
    clus_sizes = np.array(self.cluster_assignments_.sum(0)).reshape(-1)
    self.cluster_assignments_ = self.cluster_assignments_[:, clus_sizes>0]
    self.n_clusters_ = self.cluster_assignments_.shape[1]

    if (self.res_file_):
      pickle.dump(self.cluster_assignments_, open(self.res_file_, 'wb'));


def CompressDimension(X):
  colSum = np.sum(np.abs(X), axis=0)
  _, idMapping = np.where(colSum > 0.0)
  X = X[:, idMapping]
  return X, idMapping
