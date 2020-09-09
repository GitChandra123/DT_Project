#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score


# In[2]:


sonar_ds = pd.read_csv('Sonar.csv', header=1, names=range(0,61))
sonar_ds


# In[3]:


sonar_ds.head()


# In[4]:


sonar_ds = pd.DataFrame(data=sonar_ds)
sonar_ds


# In[5]:


sonar_ds.columns


# In[6]:


sonar_ds.head()


# In[7]:


sonar_ds.isnull().sum()


# In[8]:


sns.heatmap(sonar_ds.isnull())
plt.show()


# In[9]:


sonar_ds.dtypes


# In[10]:


x=sonar_ds.iloc[:,0:-1]


# In[11]:


x.shape


# In[12]:


pca = PCA(n_components=10)
x = pca.fit_transform(x)
x.shape


# In[13]:


y = sonar_ds.iloc[:,-1]
y.shape


# In[14]:


pd.DataFrame(data=x)


# In[15]:


#LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
y


# In[16]:


sns.distplot(x)
plt.show()


# In[17]:


plt.hist(x)
plt.show()


# In[18]:


#StandardScaler
scale = StandardScaler()
x = scale.fit_transform(x)
x


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .22, random_state= 43)


# In[20]:


x_train.shape


# In[21]:


y_train.shape


# In[22]:


x_test.shape


# In[23]:


y_test.shape


# In[24]:


lr = LogisticRegression()
dtc = DecisionTreeClassifier()
svc = SVC()
gnb = GaussianNB()


# In[25]:


models = []
models.append(('LogisticRegression', lr))
models.append(('DecisionTreeClassifier', dtc))
models.append(('SVC', svc))
models.append(('GaussianNB', gnb))


# In[26]:


model = []
score = []
cvs = []
rocscore = []
preds = []

for name, mod in models:
    
    print ('\n************************************************',name,'***************************')
    model.append(mod)    
    mod.fit(x_train, y_train)
    scr = mod.score(x_train, y_train)
    pred = mod.predict(x_test)
    preds.append(pred)
                       
    print ("Accuracy score: ", accuracy_score(y_test, pred)*100)
    score.append(accuracy_score(y_test, pred)*100)
    
    sc = cross_val_score(mod, x ,y, cv=10, scoring='accuracy').mean()
    print ("Cross_Val_Score = ", sc*100) 
    cvs.append(sc*100)
    print ("\n")

    fpr, tpr, thresholds = roc_curve(y_test, pred)                                    
    print (fpr, tpr, thresholds)
    
    roc_auc = auc(fpr, tpr)
    print ('RUC_AUC_Score: ', roc_auc)
    rocscore.append(roc_auc*100)
    print ("\n")
    
    cm = confusion_matrix(y_test, pred)    
    print ("\n Classification Report: \n", classification_report(y_test, pred))
    print ("\nConfusion_matrix: \n", confusion_matrix(y_test, pred))
    
    plt.figure(figsize=(10, 40))
    plt.subplot(911)
    plt.title("\n Heatmap of confusion matrix\n")
    sns.heatmap(cm, annot=True)
    
    plt.subplot(912)
    plt.title(name)
    plt.plot(fpr, tpr, label='AUC = %0.2f'%(roc_auc))
    
    plt.plot([0,1], [0,1], 'r--')
    plt.legend(loc='lower right')
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    print ('\n\n')


# In[27]:


result = pd.DataFrame({'Model': model, 'Accuracy_score': score, 'Cross_Val_Score': cvs, 'ROC_Auc_Curve': rocscore, 'Predicted Values': preds})
result


# In[28]:


from sklearn.externals import joblib
joblib.dump(svc, 'Sonar_result.joblib')
clf = joblib.load('Sonar_result.joblib')
clf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[29]:


gnb = GuassianNB()
gnb.fit


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




