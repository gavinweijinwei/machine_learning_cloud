# First let's import the dataset, using Pandas.
import sys
import pandas as pd
import numpy as np
import pickle

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#pd.options.display.encoding = sys.stdout.encoding
# make sure you're in the right directory if using iPython!
#train = pd.read_csv("D:\\workspace\\DataSets\\Redline_model_all.csv", encoding="GBK")
#test = pd.read_csv("D:\\workspace\\DataSets\\h2.csv", encoding="GBK")


#train = pd.read_csv("Redline_model_all.csv", encoding="GBK")
#test = pd.read_csv("h2.csv", encoding="GBK")
#test = pd.read_csv("h2.csv", skiprows=10751, nrows=5405, encoding="GBK")
#train = pd.read_csv("h2.csv", nrows=10750, encoding="GBK")
#train = pd.read_csv("dataset0701-0709.csv",nrows=9000, encoding="GBK")
#test = pd.read_csv("dataset0701-0709.csv", skiprows=9001, nrows=4600,encoding="GBK")
train = pd.read_csv("../training.csv", encoding="GBK")
test = pd.read_csv("../test.csv", encoding="GBK")
test.columns = train.columns

def getKS (score, bad):
    order = np.argsort(-score)
    bad_sorted = np.array([0]*len(order))
    for i in range(len(order)):
        bad_sorted[i] = bad[order[i]]

    good_sorted = 1-bad_sorted

    ks = 0.0
    for i in range(len(order)-1):
        ks = np.maximum(ks, np.abs(sum(bad_sorted[:i+1]).astype(float)/sum(bad_sorted).astype(float) - sum(good_sorted[:i+1]).astype(float)/sum(good_sorted).astype(float)))
    return ks

#attributes_cols = full_cols[1:-2]
#attributes_cols = full_cols[:-1]
full_cols = train.columns
attributes_cols = full_cols[:-1]

result_cols = (['bad'])

trainAttr = train.as_matrix(attributes_cols)
trainRes = train.as_matrix(result_cols)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

trainAttr[np.isnan(trainAttr)] = np.median(trainAttr[~np.isnan(trainAttr)])
trainRes[np.isnan(trainRes)]   = np.median(trainRes[~np.isnan(trainRes)])

# test
testArr = test.as_matrix(attributes_cols)
testArr[np.isnan(testArr)] = np.median(testArr[~np.isnan(testArr)])

badList = test.as_matrix(['bad'])

estimnatorList = np.array([1, 5])
maxfeatureList = np.array([1])
minsampleleaf  = np.array([0, 5])

estimnatorList = np.array([1, 5, 10, 30, 50, 100, 200, 500, 700, 1000, 1200, 1500])
maxfeatureList = np.array([1, 5, 10, 15, 20, 30, 50, 75, 100, 120, 150])
minsampleleaf  = np.array([1, 5, 15, 50, 75, 100, 120])


estimnatorList = np.array([1500])
maxfeatureList = np.array([150])
minsampleleaf  = np.array([75])

n = 1

for minleaf in minsampleleaf:
    for est in estimnatorList:
        for maxf in maxfeatureList:
           for i in np.arange(n):
               rfr = RandomForestRegressor(n_estimators=est, max_features=maxf, min_samples_leaf=minleaf, max_leaf_nodes=30, max_depth=30)
                #rf=rf.fit(trainAttr, trainRes)
               rfr = rfr.fit(trainAttr, trainRes.ravel())
               # Pickle the model
               model_file = "../rf.sav"
               rf_pickle = open(model_file, 'wb')
               pickle.dump(rfr, rf_pickle)
               rf_pickle.close()
               results = rfr.predict(testArr)
               file = open("output.dat", "a")
               file.write(str(est) + "," +  str(maxf) + "," + str(minleaf) + " KS -->"+ str(getKS(results, badList)) + "\n")
               file.close()
