#!/usr/bin/env python
# coding: utf-8

# #Bank customer churn prediction using Machine learning 
# Task 1 of codsoft internship

# In[1]:


import pandas as pd


# In[2]:


data= pd.read_csv("Churn_Modelling.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


print("No. of rows :-", data.shape[0])
print("No. of columns :-", data.shape[1])


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.isnull()


# In[10]:


data.columns


# In[11]:


data=data.drop(['RowNumber', 'CustomerId', 'Surname'] , axis =1)


# In[12]:


data.head()


# In[13]:


data['Geography'].unique()


# In[14]:


data = pd.get_dummies(data,drop_first=True)


# In[15]:


data.head()


# In[16]:


data['Exited'].value_counts()


# In[ ]:





# In[17]:


x=data.drop('Exited', axis=1)
y=data['Exited']


# In[18]:


x


# In[19]:


y


# Handling Imbalanced data with SMOTE

# In[20]:


from imblearn.over_sampling import SMOTE


# In[21]:


x_res,y_res = SMOTE().fit_resample(x,y)


# In[22]:


y_res.value_counts()


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split (x_res,y_res, test_size=0.20, random_state=42)


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


sc = StandardScaler()   #STANDARDISATION


# In[26]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform (x_test)


# In[27]:


x_train


# In[28]:


x_test


# # Diffrent Algorithm 
# 1 . Logistic Regrssion

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score


# In[30]:


log= LogisticRegression()


# In[31]:


log.fit(x_train,y_train)


# In[32]:


y_pred1=log.predict(x_test)


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


ac= accuracy_score(y_test,y_pred1)


# In[35]:


ac


# In[36]:


precision_score(y_test,y_pred1)


# In[37]:


recall_score(y_test,y_pred1)


# In[38]:


f1_score(y_test,y_pred1)


# 2nd method :- Support Vector Machine

# In[39]:


from sklearn import svm


# In[40]:


svm= svm.SVC ()


# In[41]:


svm.fit(x_train, y_train)


# In[42]:


y_pred2 = svm.predict(x_test)


# In[43]:


accuracy_score(y_test,y_pred2)


# In[44]:


precision_score(y_test,y_pred2)


# In[45]:


f1_score(y_test,y_pred2)


# In[46]:


recall_score(y_test,y_pred2)


# In[ ]:





# method 3:- Decision Tree Classifier

# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dt = DecisionTreeClassifier()


# In[49]:


dt.fit(x_train,y_train)


# In[50]:


y_pred4= dt.predict(x_test)


# In[51]:


accuracy_score(y_test , y_pred4)


# In[52]:


precision_score(y_test,y_pred4)


# In[53]:


f1_score(y_test,y_pred4)


# In[ ]:





# method 4:-  KNeighbors Classifier

# In[54]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:





# In[55]:


knn=  KNeighborsClassifier()


# In[56]:


knn.fit(x_train, y_train)


# In[57]:


y_pred3 = knn.predict(x_test)


# In[58]:


accuracy_score(y_test, y_pred3)


# In[59]:


f1_score(y_test, y_pred3)


# In[60]:


recall_score(y_test, y_pred3)


# In[61]:


precision_score(y_test, y_pred3)


# In[ ]:





# method 5 :- Random Forest Classifier

# In[62]:


from sklearn.ensemble import RandomForestClassifier


# In[63]:


rf=RandomForestClassifier()


# In[64]:


rf.fit(x_train,y_train)


# In[65]:


y_pred5= rf.predict(x_test)


# In[66]:


accuracy_score(y_test , y_pred5)


# In[67]:


precision_score(y_test,y_pred5)


# In[68]:


f1_score(y_test,y_pred5)


# In[69]:


recall_score(y_test,y_pred5)


# In[ ]:





# In[70]:


final_data=pd.DataFrame({'Models': ['LR','SVC','KNN','DT','RF'],
                         'ACC':[accuracy_score(y_test,y_pred1),
                                accuracy_score(y_test,y_pred2),
                                 accuracy_score(y_test,y_pred3),
                                accuracy_score(y_test,y_pred4),
                                   accuracy_score(y_test,y_pred5)
                                    ]})


# In[71]:


final_data


# In[72]:


import seaborn as sns
import matplotlib as plt


# In[73]:


sns.barplot(x='Models', y='ACC', data=final_data)


# In[75]:


final_data=pd.DataFrame({'Models': ['LR','SVC','KNN','DT','RF'],
                         'PS':[precision_score(y_test,y_pred1),
                               precision_score(y_test,y_pred2),
                                 precision_score(y_test,y_pred3),
                               precision_score(y_test,y_pred4),
                                   precision_score(y_test,y_pred5)
                                    ]})


# In[76]:


sns.barplot(x='Models', y='PS', data=final_data)


# In[ ]:





# SAVE THE MODEL
# 

# In[77]:


x_res = sc.fit_transform(x_res)


# In[78]:


rf.fit(x_res,y_res)


# In[79]:


import joblib


# In[80]:


joblib.dump(rf,'churn_prediction')


# In[81]:


model=joblib.load('churn_prediction')


# In[82]:


data.columns


# In[83]:


model.predict([[619,42,2,0,0,0,0,101348.88,0,0,0]])


# GUI

# In[84]:


from tkinter import*


# In[85]:


from sklearn.preprocessing import StandardScaler
import joblib


# In[88]:


def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=float(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=float(e8.get())
    p9=int(e9.get())
    if p9 == 1:
        Geography_Germany=1
        Geography_Spain=0
        Geography_France=0
    elif p9 == 2:
        Geography_Germany=0
        Geography_Spain=1
        Geography_France=0
    elif p9 == 3:
        Geography_Germany=0
        Geography_Spain=0
        Geography_France=1
    p10=int(e10.get())
    model = joblib.load('churn_prediction')
    result=model.predict(sc.transform([[p1,p2,p3,p4,
                           p5,p6,
                           p7,p8,Geography_Germany,Geography_Spain,p10]]))
    if result == 0:
        Label(master, text="No Exit").grid(row=31)
    else:
        Label(master, text="Exit").grid(row=31)


master = Tk()
master.title("Bank Customers Churn Prediction Using Machine Learning")


label = Label(master, text = "Customers Churn Prediction Using ML"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="CreditScore").grid(row=1)
Label(master, text="Age").grid(row=2)
Label(master, text="Tenure").grid(row=3)
Label(master, text="Balance").grid(row=4)
Label(master, text="NumOfProducts").grid(row=5)
Label(master, text="HasCrCard").grid(row=6)
Label(master, text="IsActiveMember").grid(row=7)
Label(master, text="EstimatedSalary").grid(row=8)
Label(master, text="Geography").grid(row=9)
Label(master,text="Gender").grid(row=10)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)


e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10,column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




