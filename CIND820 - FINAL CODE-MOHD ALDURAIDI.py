#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


dfo = pd.read_csv("bank-full.csv",";")
df = dfo.copy()
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


print(round(df.y.value_counts(normalize=True),2))
sns.countplot(y = 'y',data = df)


# In[ ]:


for col in df.select_dtypes(include = 'object').columns:
    print(col)
    print(df[col].unique())


# In[ ]:


cat_feat = [feature for feature in df.columns if df[feature].dtypes =='O']
for feature in cat_feat:
    print("The feature is {} and the number of categories are {}".format(feature, len(df[feature].unique())))


# In[ ]:


plt.figure(figsize=(15,88),facecolor='white')
plotnumber = 1

for categorical_feature in cat_feat:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y = categorical_feature,data = df)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()


# In[ ]:


for categorical_feature in cat_feat:
    sns.catplot(x = 'y',col = categorical_feature,kind = 'count', data = df)
plt.show()


# In[ ]:


num_feat = [feature for feature in df.columns if df[feature].dtypes !='O']
print("Number of Numerical Variable is: ",len(num_feat) )
df[num_feat].head()


# In[ ]:


disc_feat = [feature for feature in num_feat if len(df[feature].unique()) <25]
print("Number of Discrete Variable is:", len(disc_feat))


# In[ ]:


cont_feat = [feature for feature in num_feat if feature not in disc_feat+['y']]
print("Number of Continuous Variable is:", len(cont_feat))


# In[ ]:


for features in num_feat:
    df2 = df.copy()
    df2[features].hist(bins=25)
    plt.xlabel(features)
    plt.ylabel('Count')
    plt.title(features)
    plt.show()


# In[ ]:


plt.figure(figsize=(20,60),facecolor='white')
plotnumber = 1
for cont_f in cont_feat:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x='y',y=df[cont_f],data = df)
    plt.xlabel(cont_f)
    plotnumber+=1
plt.show()


# In[ ]:


plt.figure(figsize=(20,60),facecolor='white')
plotnumber = 1
for num_f in num_feat:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(df[num_f])
    plt.xlabel(num_f)
    plotnumber+=1
plt.show()


# In[ ]:


cor_mat = df.corr()
fig=plt.figure(figsize=(15,9))
sns.heatmap(cor_mat,annot = True)


# In[ ]:


df.groupby(['y','default']).size()


# In[ ]:


df.drop(['default'],axis = 1, inplace = True)


# In[ ]:


df.groupby(['y','pdays']).size()


# In[ ]:


df.drop(['pdays'],axis = 1,inplace = True)


# In[ ]:


df.groupby(['age'],sort=True)['age'].count()


# In[ ]:


df.groupby(['y','balance'],sort=True)['balance'].count()


# In[ ]:


df.groupby(['y','duration'],sort=True)['duration'].count()
df1 = df[df['duration'] < 5/60]
df1.groupby(['y','duration'],sort=True)['duration'].count()


# In[ ]:


df['duration'] = df['duration'].apply(lambda n:n/60).round(2)


# In[ ]:


df.groupby(['y','campaign'],sort=True)['campaign'].count()


# In[ ]:


df1 = df[df['campaign']<33]


# In[ ]:


df1.groupby(['y','campaign'],sort=True)['campaign'].count()


# In[ ]:


df1 = df[df['previous']<31]


# In[ ]:


from sklearn.preprocessing import StandardScaler
df = dfo.copy()
scaler = StandardScaler()
num_col = ['age', 'day', 'campaign', 'pdays','previous']
df[num_col] = scaler.fit_transform(df[num_col])
df.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False)
cat_col = ['job', 'marital', 'education', 'default', 'housing','loan', 'contact', 'month', 'poutcome']
df = dfo.copy()
df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_col]))
df_encoded.columns = encoder.get_feature_names(cat_col)
df = df.drop(cat_col,axis =1)
df = pd.concat([df_encoded,df],axis =1)
df['y'] = df['y'].apply(lambda x: 1 if x=='yes' else 0)
print("Shape of Data frame is:", df.shape)
df.head()


# In[ ]:


X = df.drop(['y'],axis =1)
y = df['y']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True,test_size=.2,random_state=1)


# In[ ]:


cols = X_train.columns
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns = [cols])
X_test = pd.DataFrame(X_test, columns = [cols])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,random_state=0)
rfc.fit(X_train, y_train)
R_pred = rfc.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
R_score = accuracy_score(y_test,R_pred)
R_cm = confusion_matrix(y_test,R_pred)
print("Random Model Score is:",(R_score.round(4))*100)
print(classification_report(y_test,R_pred))
print("Confusion Matrix for Random Forest is:",'\n', R_cm)


# In[ ]:


from sklearn.linear_model import LogisticRegression
LG_model = LogisticRegression()
Fun_LG = LG_model.fit(X_train,y_train)
L_pred = LG_model.predict(X_test)


# In[ ]:


L_score = accuracy_score(y_test,L_pred)
L_cm = confusion_matrix(y_test,L_pred)
print("Logistic Model Score is:",(L_score.round(4))*100)
print(classification_report(y_test,L_pred))
print("Confusion Matrix for Logistic Regression is:",'\n', L_cm)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 20)
K_model= knn.fit(X_train, y_train)
K_pred = K_model.predict(X_test)


# In[ ]:


K_score = accuracy_score(y_test,K_pred)
K_cm = confusion_matrix(y_test,K_pred)
print("KNN Score is:",(K_score.round(4))*100)
print(classification_report(y_test,K_pred))
print("Confusion Matrix for KNN is:",'\n', K_cm)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
DTree = clf.fit(X_train,y_train)
D_pred = DTree.predict(X_test)


# In[ ]:


D_score = accuracy_score(y_test,D_pred)
D_cm = confusion_matrix(y_test,D_pred)
print("Decision Tree Score is:",(D_score.round(4))*100)
print(classification_report(y_test,D_pred))
print("Confusion Matrix for Decision Tree is:",'\n', D_cm)

