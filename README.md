# RandProj: an Embedding based K-Nearest Neighbour Classifier for Extreme Classification
In [extreme classification](http://manikvarma.org/downloads/XC/XMLRepository.html) problems, 
the number of classes can be very large. Thus,
in those classification problems, the training time, prediction time and the model
size of a naive classifier can be infeasible. In this project, we have developed an 
embedding based K-Nearest Neighbors (KNN) algorithm to solve the above three problems. 
The algorithm has adopted two strategies towards making a KNN classifier efficient
for extreme classification:

* It uses embedding of the feature vectors into a smaller dimentional vector space. 
The KNN graph is built on this reduced dimentional feature space. 
[Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) 
allows the dimension of the reduced feature space as small as logarithmic order of the number of 
labels without losing too much information.
* It uses an approximate KNN search algorithm to find nearest neighbors in logarithmic complexity.

## Getting Started
These instructions will help you to get a copy of the project and to setup your local machine for 
development and testing purposes. 

### Prerequisites
Following packages are required to run the code

```
Python 3
joblib
numpy
scipy
sklearn
nmslib
pickle
```
To install the package, use the command:
```
pip3 install <package name>
```

### Cloning the project
To clone the project, use the command on your Ubuntu terminal

```
  git clone https://suvadeep-iitb@bitbucket.org/suvadeep-iitb/knnclassificationusingrandomprojection.git
```
After the successful completion of the command, the folder named knnclassificationusingrandomprojection
will be created. To run the test, please change your directory to knnclassificationusingrandomprojection.

## Running Test
These instructions will help you to perform experiments

### Preparing dataset
The file Toy_Example/Bibtex_data.txt contains a data in [SVMLight format](https://blog.argcv.com/articles/5371.c).
The following command reads data from Toy_Example/Bibtex_data.txt file, creates
the train and test splits in 70:30 ratio and save them in a pickle file:

```
python3 SaveDataInPickleFormat.py Toy_Example/Bibtex_data.txt 0.7 bibtex.pkl
```
After running the command, the bibtex.pkl file will be created in the same folder.

### Running Experiments
Once the bibtex.pkl has been created, to run experiments on this dataset, type
the following command:

```
python3 ExpScript.py
```

It will run the experiments using deault hype-parameter values and show you the
result on the terminal. To run experiments using other hyper-parametering
setting, you can modify the params dictionary in ExpScipt.py.


## Acknowledgments
The work has been jointly done with Prof. Ganesh Ramakrishnan, IIT Bombay, India
and Prateek Jain, Microfoft Research India.