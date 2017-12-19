import pickle
import labelCount as lc

lambdaList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

for i in [5, 14]:
    labelStruct = lc.labelStructs[i]
    
    resFilePrefix = labelStruct.resFile
    bestPrecision = [-1]*5;
    bestPrecisionTrain = [-1]*5;
    bestLambda = 0;
    for lamb in lambdaList:
      resFile = 'Results/Multiclass_'+resFilePrefix+'_TS0'+'_L'+str(lamb)+'.pkl'
      res = pickle.load(open(resFile, 'rb'))
      if bestPrecision[0] < res['testRes']['prcision'][0]:
        bestPrecision = res['testRes']['prcision']
        #bestPrecisionTrain = res['trainRes']['precision']
        bestLambda = lamb
    print('File: '+labelStruct.fileName+' Emb dim: ')
    print('Best result found for lambda: '+str(bestLambda))
    print('Precisions (Test\tTrain):')
    for i in range(5):
      print('\t'+str(bestPrecision[i])+'\t'+str(bestPrecisionTrain[i]))
    print('')
    print('')
    print('')
