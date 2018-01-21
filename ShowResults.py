import pickle
import labelCount as lc
import sys

# For the description of the hyper-paramters, please look at the file Script.py
# The values should be same as Script.py
lambdaList = [0.01, 0.1]
embDimList = [20, 50]
nnTestList = [5, 10]


if __name__ == '__main__':
    resFilePrefix = sys.argv[1]

    for embDim in embDimList:
      for nn,nnTest in enumerate(nnTestList):
        bestPrecision = [-1]*5;
        bestPrecisionTrain = [-1]*5;
        bestLambda = 0;
        for lamb in lambdaList:
          resFile = resFilePrefix+'_L'+str(lamb)+'_D'+str(embDim)+'.pkl'
          res = pickle.load(open(resFile, 'rb'))
          if bestPrecision[0] < res['testRes'][nn]['precision'][0]:
            bestPrecision = res['testRes'][nn]['precision']
            bestPrecisionTrain = res['trainRes'][nn]['precision']
            bestLambda = lamb
        print('Emb dim: '+str(embDim)+' nnTest: '+str(nnTest))
        print('Best result found for lambda: '+str(bestLambda))
        print('Precisions (Test\tTrain):')
        for i in range(5):
          print('\t'+str(bestPrecision[i][0])+'\t'+str(bestPrecisionTrain[i][0]))
        print('')
      print('')
      print('')
    print('')
    print('')
