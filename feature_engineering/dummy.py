# First let's import the dataset, using Pandas.
import sys
import pandas as pd
import numpy as np

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

total = pd.read_csv("../datasets0608_0618_with_Todun.csv", encoding="GBK")

full_cols = np.array(['grxx1043','GRXX1009','GRXX1014','GRXX1026','GRXX1041','GRXX1047','GRXX1048','DKXX1016','DKXX1017','DKXX1019','DKXX1034','DKXX1035','DKXX1037','DKXX1038','DKLX1014','DKLX1015','DKLX1020','DKLX1022','DKLX1023','DKLX1028','DKLX1029','DKLX1036','DKLX1037','DKLX1038','DKLX1039','DKLX1041','DKLX1043','DKLX1048','DKLX1019','DKLX1021','DKLX1058','DKLX1052','DXJL1001','DXJL1002','DXJL1003','DXJL1004','DXJL1005','GLXX1004','GLXX1005','GLXX1007','GLXX1008','GLXX1017','GLXX1018','GLXX1019','GLXX1021','GLXX1046','GLXX1047','GLXX1073','GLXX1074','GLXX1078','GLXX1079','GLXX1080','GLXX1086','GLXX1088','GLXX1089','GLXX1091','GLXX1092','GLXX1093','GLXX1094','GLXX1095','GLXX1096','GLXX1097','GLXX1099','TXL00A0001','TXL00C0001','TXL00B0002','TXL00B0004','TXL00B0006','TXL00D0001','TXL00A0003','TXL00B0007','TXL00B0009','TXL00A0005','TXL00A0006','TXL00C0002','TXL00B0011','TXL00B0012','TXL00B0013','TXL00B0014','TXL00B0015','TXL00B0016','NBHMD1005','SYYYS1001','SYYYS1004','SYYYS1005','SYYYS1006','SYYYS1011','SYYYS1015','SYYYS1016','SYYYS1017','SYYYS1018','SYYYS1019','SYYYS1020','SYYYS1021','SYYYS1022','SYYYS1023','SYYYS1024','SYTB1001','SYTB1012','SYTB1013','SYTB1014','SYTB1015','SYTB1016','SYTB1017','SYTB1018','SYTB1019','SYTB1022','SYTB1024','KXGRFX1002','KXGRFX1004','SYYHK1001','SYYHK1004','SYYHK1008','SYYHK1009','SYYHK1010','SYYHK1011','SYYHK1012','SYYHK1013','SYYHK1014','SYYHK1015','SYYHK1016','SYYHK1017','SYYHK1018','SYYHK1019','SYYHK1020','SYYHK1021','SYYHK1022','SYYHK1023','SYYHK1024','SYYHK1025','SYYHK1026','SYYHK1027','TDFXJC1016','TDFXJC1024','TDFXJC1025','JXLJMG1002','JXLJMG1003','JXLJMG1004','JXLJMG1005','JXLJMG1008','JXLJMG1019','JXLJMG1020','JXLJMG1021','JXLJMG1022','JXLJMG1023','JXLJMG1024','JXLJMG1025','JXLJMG1026','QHFXD1004','QHFXD1005','QHFXD1008','QHFXD1009','QHHXD1002','QHQZD1001','ZMF1002','grxx1008','GRXX1010','grxx1011','grxx1012','GRXX1016','grxx1027','grxx1028','grxx1034','grxx1035','grxx1040','GRXX1042','qhdzt1002','syyhk1002','syyhk1005','tdfxjc1002','tdfxjc1006','tdfxjc1008','tdfxjc1011','tdfxjc1014','tdfxjc1015','zmf1001','zzchmd1002','zzchmd1003','syyys1003'])
#full_cols = np.concatenate((full_cols, np.array(['var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8','var9', 'var10', 'var11', 'var12', 'var13', 'var14', 'var15','var16', 'var17', 'var18', 'var19', 'var20', 'var21', 'var22','var23', 'var24', 'var25', 'var26', 'var27', 'var28', 'var29','var30', 'var31', 'var32', 'var33', 'var34', 'var35', 'var36','var37', 'var38', 'var39', 'var40', 'var41', 'var42', 'var43','var44', 'var45', 'var46', 'var47', 'var48', 'var49', 'var50', 'var51'])))
attributes_cols = full_cols
#cate_cols = np.array(['GRXX1014'])
cate_cols = np.array([])
result_cols = (['bad'])

total_df = pd.DataFrame(data=total, columns=attributes_cols)
result_df = pd.DataFrame(data=total, columns=result_cols)

###### Get all categorical features #########
cate_cols = cate_cols.astype('S20')
for f in attributes_cols:
    #print f
    if total_df[f].unique().size < 0:
        cate_cols = np.insert(cate_cols, 0, f)
print cate_cols.size
cate_cols = np.unique(cate_cols)
print cate_cols.size
#############################################

for col in cate_cols:
    print col
    out_df = pd.get_dummies(total_df[col], prefix=col, dummy_na=True)
    total_df = pd.concat([total_df, out_df], axis=1)
    total_df = total_df.drop([col], axis=1)



train_df = total_df[:9500]
train_df = pd.concat([train_df, result_df[:9500]], axis=1)
test_df  = total_df.tail(4000)
test_df  = pd.concat([test_df, result_df.tail(4000)], axis=1)


print total_df.shape
print train_df.shape
print test_df.shape

## Input for NaN
train_df[np.isnan(train_df)] = np.median(train_df[~np.isnan(train_df)])
test_df[np.isnan(test_df)] = np.median(test_df[~np.isnan(test_df)])


## Write to CSV
train_df.to_csv("../training.csv", sep=',', index=False)
test_df.to_csv("../test.csv", sep=',', index=False)
                                                                               
