#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[80]:


df_n = df.copy()
df = pd.read_csv("bank-full.csv",";")
df.head()


# In[81]:


df.isnull().sum()


# In[82]:


df.describe()


# In[83]:


df.columns


# In[84]:


df.info()


# In[85]:


categorical_features = df[['job', 'marital', 'education', 'default','housing','loan', 'contact', 'month','poutcome', 'y']]
plt.figure(figsize=(15,88),facecolor='white')
plotnumber = 1

for categorical_feature in categorical_features:
    ax = plt.subplot(12,2,plotnumber)
    sns.countplot(y = categorical_feature,data = df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()


# In[86]:


for categorical_feature in categorical_features:
    sns.catplot(x = 'y',col = categorical_feature,kind = 'count', data = df)
plt.show()


# In[87]:


numerical_features = df[['age', 'balance','day','duration', 'campaign', 'pdays','previous']]
numerical_features.head()


# In[88]:


for features in numerical_features:
    df2 = df.copy()
    df2[features].hist(bins=25)
    plt.xlabel(features)
    plt.ylabel('Count')
    plt.title(features)
    plt.show()
    


# In[89]:


plt.figure(figsize=(20,60),facecolor='white')
plotnumber = 1
for num_feat in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x='y',y=df[num_feat],data = df)
    plt.xlabel(num_feat)
    plotnumber+=1
plt.show()


# In[90]:


plt.figure(figsize=(20,60),facecolor='white')
plotnumber = 1
for num_feat in numerical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(df[num_feat])
    plt.xlabel(num_feat)
    plotnumber+=1
plt.show()


# In[91]:


df['month'] = pd.Categorical(df['month'],["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
df.sort_values(by="month",ascending = True)


# In[92]:


df['job'] = df['job'].replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,0))
df['marital'] = df['marital'].replace(('divorced','married','single'),(1,2,3))
df['education'] = df['education'].replace(('primary','tertiary','secondary','unknown'),(1,3,2,0))
df['default'] = df['default'].replace(('yes','no'),(1,0))
df['loan'] = df['loan'].replace(('yes','no'),(1,0))
df['housing'] = df['housing'].replace(('yes','no'),(1,0))
df['poutcome'] = df['poutcome'].replace(('failure','unknown','other','success'),(0,0,1,2))
y_data = df['y']
df=df.drop(['y','contact','month'],axis = 1)


# In[93]:


df = df.apply(lambda x : (x - min(x))/(max(x) - min(x)))
df.info()


# In[94]:


df.corr()


# In[95]:


x_data = df.copy()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 100)


# In[96]:


from sklearn.linear_model import LogisticRegression
LG_model = LogisticRegression()
Fun_LG = LG_model.fit(x_train,y_train)
PRED_LG = LG_model.predict(x_test)
PRED_LG


# In[97]:


from sklearn.neighbors import KNeighborsClassifier
knn5 = KNeighborsClassifier(n_neighbors = 10)
Q10_KNN = knn5.fit(x_train, y_train)
PRED_KNN = Q10_KNN.predict(x_test)
PRED_KNN


# In[98]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
DTree = clf.fit(x_train,y_train)
PRED_DTree = DTree.predict(x_test)
PRED_DTree


# In[99]:


from sklearn.metrics import confusion_matrix
confusion_m_LG = confusion_matrix(y_test,PRED_LG)
confusion_m_KNN = confusion_matrix(y_test,PRED_KNN)
confusion_m_CLF = confusion_matrix(y_test,PRED_DTree)

print('Confusion Matrix for Logistic Regression is:','\n', confusion_m_LG)
print('Confusion Matrix for KNN Regression is:','\n',confusion_m_KNN)
print('Confusion Matrix for DecistionTree Regression is:','\n',confusion_m_CLF)


# In[100]:


from sklearn.metrics import accuracy_score, precision_score
print('Accuracy = (TP+TN)/(TP+TN+FP+FN)','\n','Precision = (TP)/(TP+FP)', '\n','Recall = (TP)/(TP+FN)')
print('Logistic Regression Accuracy = ',round(accuracy_score(y_test,PRED_LG),4))
print('Logistic Regression Precision = ',round(confusion_m_LG[0,0]/(confusion_m_LG[0,0]+confusion_m_LG[1,0]),4))
print('Logistic Regression Recall = ',round(confusion_m_LG[0,0]/(confusion_m_LG[0,0]+confusion_m_LG[0,1]),4))
print('KNN Regression Accuracy = ',round(accuracy_score(y_test,PRED_KNN),4))
print('KNN Regression Precision = ',round(confusion_m_KNN[0,0]/(confusion_m_KNN[0,0]+confusion_m_KNN[1,0]),4))
print('KNN Regression Recall = ',round(confusion_m_KNN[0,0]/(confusion_m_KNN[0,0]+confusion_m_KNN[0,1]),4))
print('DTree Regression Accuracy = ',round(accuracy_score(y_test,PRED_DTree),4))
print('DTree Regression Precision = ',round(confusion_m_CLF[0,0]/(confusion_m_CLF[0,0]+confusion_m_CLF[1,0]),4))
print('DTree Regression Recall = ',round(confusion_m_CLF[0,0]/(confusion_m_CLF[0,0]+confusion_m_CLF[0,1]),4))


# In[101]:


from sklearn.metrics import classification_report
print('Logistic Regression = ','\n',classification_report(y_test, PRED_LG))
print('KNN Regression = ','\n',classification_report(y_test, PRED_KNN))
print('DTree = ','\n',classification_report(y_test, PRED_DTree))

