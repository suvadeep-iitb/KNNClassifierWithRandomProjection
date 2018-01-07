import pickle
import labelCount as lc

mu1List = [1]
mu2List = [1]
mu3List = [1]
mu4List = [0]
nnTestList = [3]
embDimList = [20]
outerIterList = [3]
maxTS = [0]

for i in [5]:
  labelStruct = lc.labelStructs[i]
  resFilePrefix = labelStruct.resFile
  for ts in maxTS:
    for ed in embDimList:
      for it in outerIterList:
        for nn,nnTest in enumerate(nnTestList):
          bestPrecision = [-1]*5;
          bestPrecisionTrain = [-1]*5;
          bestMu1 = 0
          bestMu2 = 0
          bestMu3 = 0
          bestMu4 = 0
          for mu1 in mu1List:
            for mu2 in mu2List:
              for mu3 in mu3List:
                for mu4 in mu4List:
                  resFile = 'Results/MOBCAP_'+resFilePrefix+'_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)+'.pkl'
                  res = pickle.load(open(resFile, 'rb'))
                  if bestPrecision[0] < res['testRes'][nn]['precision'][0]:
                    bestPrecision = res['testRes'][nn]['precision']
                    #bestPrecisionTrain = res['trainRes'][nn]['precision']
                    bestMu1 = mu1
                    bestMu2 = mu2
                    bestMu3 = mu3
                    bestMu4 = mu4
          print('File: '+labelStruct.fileName+' Emb dim: '+str(ed)+' nnTest: '+str(nnTest)+' Iter: '+str(it))
          print('Best result found for mu1: '+str(bestMu1)+' mu2: '+str(bestMu2)+' mu3: '+str(bestMu3)+' mu4: '+str(bestMu4))
          print('Precisions (Test\tTrain):')
          for i in range(5):
            print('\t'+str(bestPrecision[i])+'\t'+str(bestPrecisionTrain[i]))
          print('')
      print('')
      print('')
  print('')
  print('')
