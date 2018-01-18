import pickle
import labelCount as lc

lambdaList = [0.0001, 0.001, 0.01, 0.1]
nnTestList = [10]
embDimList = [20, 50]

for i in [18]:
    labelStruct = lc.labelStructs[i]
    
    resFilePrefix = labelStruct.resFile
    for embDim in embDimList:
      for nn,nnTest in enumerate(nnTestList):
        bestPrecision = [-1]*5;
        bestPrecisionTrain = [-1]*5;
        bestLambda = 0;
        for lamb in lambdaList:
          resFile = 'Results/RandProj_'+resFilePrefix+'_TS0'+'_L'+str(lamb)+'_D'+str(embDim)+'.pkl'
          res = pickle.load(open(resFile, 'rb'))
          if bestPrecision[0] < res['testRes'][nn]['precision'][0]:
            bestPrecision = res['testRes'][nn]['precision']
            #bestPrecisionTrain = res['trainRes'][nn]['precision']
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
