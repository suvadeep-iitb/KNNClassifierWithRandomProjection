import pickle
import labelCount as lc

lambdaList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
nnTestList = [5, 10, 20]
embDimList = [10, 20]
numClustersList = [1]

for i in [1]:
    labelStruct = lc.labelStructs[i]
    
    resFilePrefix = labelStruct.resFile
    for ed in embDimList:
      for nc in numClustersList:
        for nn,nnTest in enumerate(nnTestList):
          bestPrecision = [-1]*5;
          bestPrecisionTrain = [-1]*5;
          bestLambda = 0;
          for lamb in lambdaList:
            resFile = 'Results/ClusteredRandProj_'+resFilePrefix+'_TS0'+'_CL'+str(nc)+'_L'+str(lamb)+'_D'+str(ed)+'.pkl'
            res = pickle.load(open(resFile, 'rb'))
            if bestPrecision[0] < res['testRes'][nn]['precision'][0]:
              bestPrecision = res['testRes'][nn]['precision']
              #bestPrecisionTrain = res['trainRes'][nn]['precision']
              bestLambda = lamb
          print('File: '+labelStruct.fileName+' Emb dim: '+str(ed)+'# of clusters: '+str(nc)+' nnTest: '+str(nnTest))
          print('Best result found for lambda: '+str(bestLambda))
          print('Precisions (Test\tTrain):')
          for i in range(5):
            print('\t'+str(bestPrecision[i])+'\t'+str(bestPrecisionTrain[i]))
          print('')
        print('')
      print('')
    print('')
    print('')