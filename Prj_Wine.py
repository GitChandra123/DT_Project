#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
#from sklearn import preprocessing


# In[ ]:





# In[61]:


wine_ds = pd.read_csv('winedataset.csv')
wine_ds


# In[62]:


wine_ds['Class'].value_counts()


# In[63]:


wine_ds.isnull().sum()


# In[64]:


wine_ds.corr()


# In[65]:


sns.heatmap(wine_ds.corr(), annot=True)
plt.show()


# In[66]:


wine_ds.describe()


# In[67]:


wine_ds.dtypes


# In[68]:


wine_ds.skew()


# In[ ]:





# In[69]:


sns.distplot(wine_ds, kde=True)
plt.show()


# In[70]:


plt.hist(wine_ds)
plt.show()


# In[71]:


# Remove skewness 
for col in wine_ds.columns:
    if wine_ds.skew().loc[col] > 0.55:
        wine_ds[col] = np.log1p(wine_ds[col])


# In[72]:


wine_ds.skew()


# In[73]:


plt.hist(wine_ds)
plt.show()


# In[74]:


# Remove skewness 
for col in wine_ds.columns:
    if wine_ds.skew().loc[col] > 0.55:
        wine_ds[col] = np.log1p(wine_ds[col])


# In[75]:


wine_ds.skew()


# In[76]:


sns.pairplot(wine_ds)
plt.show()


# In[77]:


from scipy.stats import zscore
z_score = abs(zscore(wine_ds))
print (wine_ds.shape)

wine_ds_final = wine_ds.loc[(z_score < 3).all(axis=1)]
print (wine_ds_final.shape)


# In[78]:


#Separatomg target and input variables

df_x = wine_ds_final.drop(columns=['Class'])
y = wine_ds_final['Class']


# In[79]:


df_x


# In[80]:


y


# In[81]:


# Scaling the input variables
# Liner Algorithm feature requires that all varibales should be on common scale


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(df_x)
x = pd.DataFrame(x, columns = df_x.columns)


# In[82]:


x.skew()


# In[ ]:





# In[83]:


x.shape


# In[84]:


y.shape


# In[85]:


from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = .22)


# In[86]:


x_train.shape


# In[87]:


y_train.shape


# In[88]:


x_test.shape


# In[89]:


y_test.shape


# In[90]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# In[91]:


KNN = KNeighborsClassifier()
sv = SVC()
DT = DecisionTreeClassifier(random_state = 6)


# In[ ]:





# In[92]:


models = []
models.append(('KNeighborsClassifier', KNN))
models.append(('SVC', sv))
models.append(('DecisionTreeClassifier', DT))


# In[93]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc


# In[107]:


Model = []
score = []
cvs = []
rocscore = []

for name, model in models:
    
    print ('***************************',name,'****************************')
    print ('\n')
    
    Model.append(name)
    model.fit(x_train, y_train)
    print (model)
    pre = model.predict(x_test)
    print ('\n')
    
    AS = accuracy_score(y_test, pre)
    print ("Accuracy score: ", AS)
    score.append(AS*100)
    print ("\n")
    
    sc = cross_val_score(model, x ,y, cv=10, scoring='accuracy').mean()
    print ("Cross_Val_Score = ", sc) 
    cvs.append(sc*100)
    print ("\n")
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pre, pos_label=1)
    print (false_positive_rate, true_positive_rate, thresholds)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print ('RUC_AUC_Score: ', roc_auc)
    rocscore.append(roc_auc*100)
    print ("\n")

    
    print ("classification_report:\n ", classification_report(y_test, pre))
    print ("\n")
    cm = confusion_matrix(y_test, pre)
    print ('\nConfusion matrix: ', confusion_matrix(y_test, pre))
    print ("\n")
    
    plt.figure(figsize=(10, 40))
    plt.subplot(911)
    plt.title("\n Heatmap of confusion matrix\n")
    sns.heatmap(cm, annot=True)
    
    plt.subplot(912)
    plt.title(name)
    plt.plot(false_positive_rate, true_positive_rate, label='AUC = %0.2f'%(roc_auc))
    plt.plot([0,1], [0,1], 'r--')
    plt.legend(loc='lower right')
    plt.ylabel('true positive rate')
    plt.xlabel('false positive rate')
    print ('\n\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[95]:


result = pd.DataFrame({'Model': Model, 'Accuracy_score': score, 'Cross_Val_Score': cvs, 'ROC_Auc_Curve': rocscore})
result


# In[96]:


from sklearn.externals import joblib


# In[97]:


from joblib import dump, load


# In[102]:


dump(sv, 'Wine_result.joblib')


# In[103]:


clf = load('Wine_result.joblib')


# In[104]:


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




