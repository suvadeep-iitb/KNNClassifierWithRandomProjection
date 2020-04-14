# RandProj: an Embedding based K-Nearest Neighbour Classifier for Extreme Classification
In [extreme classification](http://manikvarma.org/downloads/XC/XMLRepository.html) problems, 
the number of classes can be very large. Thus,
in those classification problems, the training time, prediction time and the model
size of a naive classifier can be very large. In this project, we develop a 
embedding based K-Nearest Neighbors (KNN) algorithm which solve the above three problems. 
The algorithm has adopted two strategies towards making a KNN classifier efficient
for the extreme classification:
- It has used embedding of the feature vectors into a smaller dimentional vector space.
The KNN graph is built on this reduced dimentional feature space. 
[Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
allows the dimension of the reduced feature space as small as logarithmic order of the number
of labels without losing too much information.
- It has used an approximate KNN search algorithm to find nearest neighbors in logarithmic
complexity.


