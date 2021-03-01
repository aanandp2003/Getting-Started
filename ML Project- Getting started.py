#!/usr/bin/env python
# coding: utf-8

# In[38]:


import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(sys.version))
import pandas
import matplotlib
import sklearn


# In[36]:



from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[14]:


url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names =['sepal-length', 'sepal-width', 'petal-length','class']
dataset = read_csv(url, names=names)


# In[15]:


print(dataset.shape)


# In[12]:


print(dataset.head(20))


# In[16]:


print(dataset.describe())


# In[17]:


print(dataset.groupby('class').size())


# In[18]:


dataset.plot(kind='box', subplots =True, layout=(2,2), sharex =False, sharey=False)
pyplot.show()


# In[19]:


scatter_matrix(dataset)
pyplot.show()


# In[31]:


array = dataset.values
X =array[:, 0:4]
y= array[:, 0:4]
X_train, X_valudation, Y_train, Y_validation = train_test_split(X,y, test_size=0.2, random_state=1)


# In[32]:


models =[]
models.append(('LR', LogisticRegression(solver ='liblinear',multi_class ='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[39]:


results =[]
names=[]
for name,model in models:
    kfold = StratifiedKFold(n_splits =10,random_state=1)
    cv_results= cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)'% (name, cv_results.mean(), cv_results.std()))


# In[33]:


pyplot.boxplot(results, labels =names)
pyplot.title('Algorithm Comapirison')
pyplot.show()


# In[ ]:




