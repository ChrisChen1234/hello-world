import numpy as np
import csv
import pandas as pd
from sklearn import svm

data_df = pd.read_csv("train_svm.csv").drop(columns=['id','species'])

data_df = np.array(data_df.values)

trainSet = np.zeros((99*7,192))
label    = np.zeros(99*7)

testSet  = np.zeros((99*3,192))
count = 1
countTrain = 0
countTest = 0
for i in range(0,99*10):
    if count == 11:
        count=1
    if count<=7:
        trainSet[countTrain] = data_df[i]
        countTrain = countTrain+1
    else:
        testSet[countTest] = data_df[i]
        countTest = countTest + 1
    count = count+1

for i in range(0,99*7):
    label[i] = int(i/7)


svmClassify = svm.SVC()
svmClassify.kernel = 'linear'
#svmClassify.gamma = 1.0
svmClassify.C = 1.0
svmClassify.fit(trainSet,label)

classifyResult = svmClassify.predict(testSet[:])

trueAns = np.zeros(99*3)
for i in range(0,99*3):
    trueAns[i] = int(i/3)
# print(trueAns)

meetCount = 0


for i in range(0,99*3):
    # print(trueAns[i])
    # print(classifyResult[i])
    if trueAns[i] ==classifyResult[i]:
         meetCount = meetCount+1

print('Classify Result:')
print(classifyResult)

print('True Answer:')
print(trueAns)

print('Meet Number =',meetCount)
print('Accuracy =',meetCount/(99*3)*100,'%')
