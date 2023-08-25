#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
plt.style.use(['seaborn-bright','dark_background'])


# # Import dataset

# In[18]:


data = pd.read_csv('churn_prediction_simple.csv')
data.head()


# In[7]:


data = data.dropna() #for dropping missing values 
data.info()


# In[9]:


#checking the dataset distribution
data['churn'].value_counts()/len(data)


# In[12]:


#separating dependent and independent variables
X=data.drop(columns=['churn','customer_id'])
Y=data['churn']


# In[14]:


#scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[17]:


#splitting the dataset 
from sklearn.model_selection import train_test_split as tts 
x_train,x_test,y_train,y_test = tts(scaled_X,Y,train_size=0.80,stratify=Y)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# # Model Building Predictions and odds Ratio 
# 

# In[20]:


from sklearn.linear_model import LogisticRegression as LR
classifier = LR(class_weight = 'balanced') #to emphasize the class es depending upon the number of observations presesnt in the data


# In[22]:


classifier.fit(x_train,y_train)
predicted_values = classifier.predict(x_test)
predicted_probabilities = classifier.predict_proba(x_test)


# In[24]:


predicted_values


# In[26]:


predicted_probabilities,predicted_probabilities.shape


# In[28]:


#Accuracy
classifier.score(x_test,y_test)


# In[30]:


#precision 
from sklearn.metrics import precision_score
precision = precision_score(y_test,predicted_values)
precision


# 38 percent of the observation predicted is false positive 

# In[32]:


#calculating recall
from sklearn.metrics import recall_score 
Recall = recall_score(y_test,predicted_values)
Recall


# it means out of all the actual positive observation only 66 percent of the observation have been predicted as positive 

# choosing between precision and recall is a matter of buisness problem 

# In[36]:


#manually calculating the F1
F1=2/((1/precision)+(1/Recall))
F1


# In[40]:


#calculating the F1-Score
from sklearn.metrics import f1_score
F1 = f1_score(y_test,predicted_values)
F1


# our model is not a good model because the value of F1 is less than 0.5 and for that now we are going to calculate precision,recall, f1_score and support at once

# In[45]:


from sklearn.metrics import precision_recall_fscore_support as PRF_summary
precision,recall,fscore,support = PRF_summary(y_test,predicted_values)


# In[47]:


precision


# In[50]:


recall


# In[52]:


fscore


# In[54]:


support

now we are getting two values because precision_recall_fscore_support,
it supports both of the classes class 0 and class 1
first for class 0 and second for class 1 
# In[56]:


from sklearn.metrics import classification_report
k = classification_report(y_test,predicted_values)
print(k)


# # Precision Recall Curve

# In[62]:


#gathering precision/recall scores for different thresholds
from sklearn.metrics import precision_recall_curve
precision_points,recall_points,threshold_points = precision_recall_curve(y_test,predicted_probabilities[:,1])
precision_points.shape,recall_points.shape,threshold_points.shape


# In[65]:


plt.figure(figsize = (7,5),dpi = 100)
plt.plot(threshold_points,precision_points[:-1],color='green',label='Precision')
plt.plot(threshold_points,recall_points[:-1],color='red',label='Recall')
plt.xlabel('Threshold_points',fontsize=15)
plt.ylabel('Score',fontsize=15)
plt.title('Precision_recall tradeoff',fontsize=20)
plt.legend()


# precision of the model increases the recall of the model decreases although the performance is not great we can infer the intersection point is somewhere near by 0.55 threshold

# # AUC-ROC Curve

# In[69]:


from sklearn.metrics import roc_curve, roc_auc_score #roc_curve returns the list of fpr and tprvalues for different valuesof the probability thresholds
fpr,tpr,threshold = roc_curve(y_test,predicted_probabilities[:,1]) #passing the probilities of class 1


# In[71]:


plt.figure(figsize=(7,5),dpi=100)
plt.plot(fpr,tpr,color='green')
plt.plot([0,1],[0,1],label='baseline',color='red')
plt.xlabel('FPR',fontsize=15)
plt.ylabel('TPR',fontsize=15)
plt.title('AUC-ROC',fontsize=20)
plt.show()
roc_auc_score(y_test,predicted_probabilities[:,1])


# # Coefficient plot

# In[73]:


#arranging the data 
c = classifier.coef_.reshape(-1)
x=X.columns
coeff_plot = pd.DataFrame({'coefficients':c,'variable':x})
#sorting the values
coeff_plot = coeff_plot.sort_values(by='coefficients')
coeff_plot.head()


# In[76]:


plt.figure(figsize=(8,6),dpi=120)
plt.barh(coeff_plot['variable'],coeff_plot['coefficients'])
plt.xlabel('coefficient_magnitude',fontsize=15)
plt.ylabel('variables',fontsize=15)
plt.title('coeff_plot',fontsize=20)


# In[ ]:




