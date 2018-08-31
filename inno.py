import urllib2
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def convertToText(html):
    soup = BeautifulSoup(html)
    relevent=soup.find_all(['p'])
    fin=""
    for r in relevent:
        fin=fin+str(r)
    return fin

data=pd.read_csv("train/train.csv")
datalen=len(data)
fixedsize=50
categList={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
for index, row in data.iterrows():
    if(row['Tag']=="news"):
        categList["news"].append(row['Webpage_id'])
    if(row['Tag']=="clinicalTrials"):
        categList["clinicalTrials"].append(row['Webpage_id'])
    if(row['Tag']=="publication"):
        categList["publication"].append(row['Webpage_id'])
    if(row['Tag']=="guidelines"):
        categList["guidelines"].append(row['Webpage_id'])
    if(row['Tag']=="forum"):
        categList["forum"].append(row['Webpage_id'])
    if(row['Tag']=="profile"):
        categList["profile"].append(row['Webpage_id'])
    if(row['Tag']=="conferences"):
        categList["conferences"].append(row['Webpage_id'])
    if(row['Tag']=="thesis"):
        categList["thesis"].append(row['Webpage_id'])
    if(row['Tag']=="others"):
        categList["others"].append(row['Webpage_id'])       
for key, value in categList.iteritems():
    totake=(len(value)*fixedsize)/datalen
    value=random.sample(value,totake)
    categList[key]=value

traindata={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
testdata={"news":[],
          "clinicalTrials":[],
           "publication":[],
           "guidelines":[],
           "forum":[],
           "profile":[],
           "conferences":[],
           "thesis":[],
           "others":[],
          }
for key, val in categList.iteritems():
    for index in range(len(val)):
        if(index%3==0):
            testdata[key].append(val[index])
        else:
            traindata[key].append(val[index])


traindatalist=[]
testdatalist=[]
for key, val in traindata.iteritems():
    for index in range(len(val)):
        traindatalist.append(val[index])
for key, val in testdata.iteritems():
    for index in range(len(val)):
        testdatalist.append(val[index])




targets=[]
for index,row in data.iterrows():
    if row['Webpage_id'] in traindatalist:
        targets.append(row['Tag'] )
test_targets=[]
for index,row in data.iterrows():
    if row['Webpage_id'] in testdatalist:
        test_targets.append(row['Tag'])


trainlistmin=[]
for i in range(79345):
    if i+1 not in traindatalist:
        trainlistmin.append(i+1)



testlistmin=[]
for i in range(79345):
    if i+1 not in testdatalist:
        testlistmin.append(i+1)


skip=sorted(trainlistmin)
fulltraindata=pd.read_csv("train/html_data.csv",skiprows=skip)

skip=sorted(testlistmin)
fulltestdata=pd.read_csv("train/html_data.csv",skiprows=skip)


dataList=[]
for html in fulltraindata['Html']:
    dataList.append(convertToText(html))


testList=[]
for html in fulltestdata['Html']:
    testList.append(convertToText(html))




count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataList)
print X_train_counts.shape


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape


clf = MultinomialNB().fit(X_train_tfidf, targets)


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(dataList, targets)


predicted = text_clf.predict(testList)
print np.mean(predicted == test_targets)


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(dataList, targets)
predicted_svm = text_clf_svm.predict(testList)
print np.mean(predicted_svm == test_targets)


preddata=pd.read_csv("train/test_nvPHrOx.csv")
preddoc=preddata['Webpage_id'].tolist()

predmin=[]
for i in range(79345):
    if i+1 not in preddoc:
        predmin.append(i+1)

skip=sorted(predmin)

print "started"

fullpreddata=pd.read_csv("train/html_data.csv",skiprows=skip)

print "finished"

predList=[]
for html in fullpreddata['Html']:
    predList.append(convertToText(html))

predicted_svm_pred = text_clf_svm.predict(predList)

print predicted_svm_pred
