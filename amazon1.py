import pandas as pd
br=pd.read_csv("C:/Users/ELCOT/Desktop/amazon 14gb/balanced_reviews.csv")




br.dropna(inplace=True)
 
br=br[br["overall"]!=3] 

over=br["overall"]
br["class"]=" "
import numpy as np
br["class"]=np.where(br["overall"]>3,1,0)

feature=br["reviewText"].values  
labels=br["class"].values


import nltk
import re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
stp=stopwords.words("english")
ps=PorterStemmer() 
review=feature[0]
features=[]
for i in range(0,feature.shape[0]):
    review=re.sub("[^a-zA-z]"," ",feature[i]) 
    review=review.lower()
    review=review.split()
    review=[word for word in review if word not in stp]
    
    review=[ps.stem(i) for i in review]
    review="".join(review)
    features.append(review)

print(features) 
 
from sklearn.model_selection import train_test_split

feature_train,feature_test,labels_train,labels_text=train_test_split(features,labels,random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

uni_nam=CountVectorizer().fit(feature_train)
len(uni_nam.get_feature_names()) 

ui_nam1=uni_nam.transform(feature_train)

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()


lr.fit(ui_nam1,labels_train)

lr.predict(uni_nam.transform(feature_test)) 

from sklearn.metrics import roc_auc_score

score=roc_auc_score(lr.predict(uni_nam.transform(feature_test)),labels_text )




#VERSION 2


from sklearn.model_selection import train_test_split
feature_train,feature_test,labels_train,labels_text=train_test_split(feature,labels,random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(min_df=5).fit(feature_train)

tfidf_vecto=tfidf.transform(feature_train) 

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(tfidf_vecto,labels_train) 

pred=lr.predict(tfidf.transform(feature_test))
v=pd.DataFrame(zip(pred,labels_text),columns=["pred","real"])

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred,labels_text) 
from sklearn.metrics import roc_auc_score
ras=roc_auc_score(pred,labels_text)


#version 3
'''
from sklearn.neighbors import KNeighborsClassifier
kn= KNeighborsClassifier(n_neighbors=5,p=1)
kn.fit(tfidf_vecto,labels_train)

pred=kn.predict(tfidf.transform(feature_test))
v=pd.DataFrame(zip(pred,labels_text),columns=["pred","real"])

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred,labels_text) 
from sklearn.metrics import roc_auc_score 
ras=roc_auc_score(pred,labels_text)
'''

import pickle
open("model.pkl","wb")
pickle.dump(lr,open("model.pkl","wb"))


pickle.dump(tfidf.vocabulary_,open("vocab_file.pkl","wb")) 



def check_pos_or_neg(text):
    re_mod=pickle.load(open("model.pkl","rb"))  
    
    vocab=pickle.load(open("vocab_file.pkl","rb")) 
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    transform=TfidfVectorizer(vocabulary=vocab)
    return re_mod.predict(transform.fit_transform([text])) 

    if (re_mod.predict(transform.fit_transform([text])))==0:
        print("negative")
    if(re_mod.predict(transform.fit_transform([text])))==1:
        print("positive")

check_pos_or_neg("its  bad")    



