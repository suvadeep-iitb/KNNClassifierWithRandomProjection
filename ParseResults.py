import pickle
import labelCount as lc

lambdaList = [0.01, 1, 100, 10000]
nnTestList = [5, 10, 20]
embDimList = [100, 500]

for i in [8, 9]:
    labelStruct = lc.labelStructs[i]
    
    resFilePrefix = labelStruct.resFile
    for embDim in embDimList:
      for nnTest in nnTestList:
        bestPrecision = [-1]*5;
        bestPrecisionTrain = [-1]*5;
        bestLambda = 0;
        for lamb in lambdaList:
          resFile = 'Results/'+resFilePrefix+'_L'+str(lamb)+'_D'+str(embDim)+'_NN'+str(nnTest)+'.pkl'
          res = pickle.load(open(resFile, 'rb'))
          if bestPrecision[0] < res['precision'][0]:
            bestPrecision = res['precision']
            bestPrecisionTrain = res['precision_tr']
            bestLambda = lamb
        print('File: '+labelStruct.fileName+' Emb dim: '+str(embDim)+' nnTest: '+str(nnTest))
        print('Best result found for lambda: '+str(bestLambda))
        print('Precisions (Test\tTrain):')
        for i in range(5):
          print('\t'+str(bestPrecision[i])+'\t'+str(bestPrecisionTrain[i]))
        print('')
      print('')
      print('')
    print('')
    print('')
