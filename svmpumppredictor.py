import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold as StratK
import sys

def pumpornot_score(C_pump,C_nopump,gamma,data,colors,alpha=1/2,k=5):
    # data (ndarray rank 2) is the full set of trainining vectors without labels
    # labels (ndarray rank 1) are the labels of the vectors used for stratified k-fold cross validation (0 must correspond to nopump)
    svc=SVC(C=1,kernel='rbf',gamma=gamma,class_weight={1:C_pump,0:C_nopump},verbose=False)
    skf=StratK(n_splits=k)
    kfolds=skf.split(data,colors)
    pumpornot_labels=np.ascontiguousarray(np.fromiter(map(lambda x: 0*(0==x)+1*(0!=x),colors),float))
    #print('pumpornot_labels:\n',pumpornot_labels)
    scores=[]
    precisions=[]
    sensitivities=[]
    ogscores=[]
    for foldno,(train,test) in enumerate(kfolds,start=1):
        #print('Calculating score on fold ',str(foldno),' of ',str(k))
        train_labels=pumpornot_labels[train]
        #print('number of pumps in train set: ', np.sum(train_labels))
        test_labels=pumpornot_labels[test]
        #print('test_labels: ',test_labels)
        #print('number of pumps in test set: ', np.sum(test_labels))
        svc.fit(data[train],train_labels)
        predicted_labels=svc.predict(data[test])
        #print('\npredicted labels :',predicted_labels)
        #print('number of predicted pumps :',np.sum(predicted_labels))
        decision_vals=svc.decision_function(data[test])
        #print('decision values: ',decision_vals)
        p=np.sum(test_labels)
        tp,fp=tuple(map(lambda tf: np.sum(np.fromiter((min(x,1) for i,x in enumerate(decision_vals) if test_labels[i]==tf and x>0),float)),[1,0]))
        fn=p-tp
        #print('[tp, fp, fn, p]: ',[tp, fp, fn, p])
        pr,se=(0 if tp+fp==0 else tp/(tp+fp),tp/p)
        #ogscore=1/(alpha/pr+(1-alpha)/se)
        #print('precision: ',pr,'\nsensitiviy: ',se)
        #print('original fscore: ',ogscore,'\n')
        precisions.append(pr)
        sensitivities.append(se)
        scores.append(-tp/(p+alpha*(fp-fn)))
        #ogscores.append(ogscore)
    print('scores: ',scores)
    return list(map(lambda x: np.sum(np.array(x))/k,[precisions,sensitivities,scores]))
    
vectors=pd.read_csv('/home/act/Desktop/SVM/SVM Python/_totalVector6969.csv',sep='\t',index_col=0)
vectors=vectors.astype({'0':'category','1':'category',vectors.columns[-1]:'category'})
pump_labels=np.ascontiguousarray(pd.read_csv('/home/act/Desktop/appspack/examples/1506/5_labelfile.csv',sep=',',index_col=0),dtype=float).flatten()

data=np.ascontiguousarray(vectors.iloc[:,3:-1],dtype=float)
labels=np.ascontiguousarray(vectors.iloc[:,-1],dtype=float)
colors=np.zeros(len(labels))
colors[np.where(labels!=0)]=np.array(pump_labels)+1
print("argv0: ",sys.argv[0])
print("argv1: ",sys.argv[1])
print("argv2: ",sys.argv[2])
with open(sys.argv[2],mode='r',encoding = "ISO-8859-1") as f:
    params=[float(x.strip()) for x in f.readlines()][1:]

with open(sys.argv[3],mode='w',encoding = "ISO-8859-1") as f:
    print("params to be written: ",params)
    f.write(str(pumpornot_score(*params,data,colors,alpha=0.8)[-1]))
#encoding = "ISO-8859-1"