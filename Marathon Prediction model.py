#!/usr/bin/env python
# coding: utf-8

# The data below reads as follows:
#     
#     id corresponds to the index of each athlete
#     
#     Category corresponds to the sex and age bracket each runner falls in (i.e MAM = Male under 40 years, 
#     WAM = Woman under 40 years, M40 = Male athlete between 40 and 45 years)
#     
#     km4week is the total number of kilometers run in the last 4 weeks before the marathon, marathon included.
#     If, for example, the km4week is 100, the athlete has run 400 km in the four weeks before the marathon
# 
#     sp4week is the average speed of the athlete in the last 4 training weeks. The average counts all the kilometers
#     done, included the slow kilometers done before and after the training. The average of the speed is 
#     this number, and with time this is one of the numbers that has to be refined
#     
#     cross training: If the runner is also a cyclist, or a triathlete. How would training other disciplines affect 
#     your timing?
# 
#     Wall21: In decimal. The tricky field. To acknowledge a good performance, as a marathoner, I have to run the 
#     first half marathon with the same split of the second half. If, for example, I run the first half marathon 
#     in 1h30m, I must finish the marathon in 3h (for doing a good job). If I finish in 3h20m, I started too fast
#     and I hit "the wall". My training history is, therefore, less valid, since I was not estimating my result
# 
#     Marathon time: In decimal. This is the final result. Based on the training history, the marathon timing must 
#     be predicted.
# 

# In[74]:


import pandas as pd

running_data = pd.read_csv('MarathonData.csv')
running_data.drop(['id', 'Wall21', 'CATEGORY'], axis=1, inplace=True)

running_data


# In[75]:


running_data.describe().T


# In[76]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[77]:


# Lets see the box plot
# It is very useful to handle the ouliers

# Lets create a box-plot of Category and MarathonTIme
plt.figure(figsize=(8,8))
sns.boxplot(x='Category',y='MarathonTime',data=running_data)
plt.show()


# The box-plot above of marathon timing and the category corresponding to the sex and age bracket each runner falls in provides the model with crucial information. The variation in marathon timing between the categories is significant and plays a role in training the model.

# In[78]:


# Lets see the scatter plot
# Lets create scatterplot of sp4week and MarathonTime

plt.figure(figsize=(8,8))
sns.scatterplot(x='sp4week',y='MarathonTime',data=running_data)
plt.show()


# In[79]:


running_data["sp4week"].mode()


# A clear outlier is present in the scatter plot for average speed and marathon timing illustrated above. We will deal with this by writing a function with the help of z-score to detect the outlier in our data and deal with it accordingly.

# In[80]:


# Detect outliers present in the dataframe
import numpy as np
import pandas as pd
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(running_data.index[running_data['sp4week'] == y])
    return outliers


# We now pass the dataset column that needs to be dealt with as 
# an input argument to the detect_outlier function

outliers_datapoints = detect_outlier(running_data["sp4week"])

print(outliers_datapoints)


# In[82]:


# Remove outliers from the dataframe
for outlier in outliers_datapoints:
    running_data.drop(outlier, inplace = True)
running_data = running_data.reset_index(drop=True)
running_data


# In[83]:


# Lets see the scatter plot
# Lets create scatterplot of sp4week and MarathonTime

plt.figure(figsize=(8,8))
sns.scatterplot(x='sp4week',y='MarathonTime',data=running_data)
plt.show()


# After we cleaned the above data, it is evident that there is a relationship between an athletes average speed and their marathon timing. There exists a negative correlation between marathon timing and average speed. The higher an athletes average speed, the lower their marathon timing. 

# In[84]:


# Lets see the scatter plot
# Lets create scatterplot of km4week and MarathonTime

plt.figure(figsize=(8,8))
sns.scatterplot(x='km4week',y='MarathonTime',data=running_data)
plt.show()


# There is a negative correlation between an athletes total kms run and their marathon timing. The more an athlete trained in the 4 weeks prior to the marathon, the quicker they managed to run in their marathon.

# In[85]:


# Lets see the box plot
# It is very useful to handle the ouliers

# Lets create box-plot of CrossTraining and MarathonTIme
plt.figure(figsize=(8,8))
sns.boxplot(x='CrossTraining',y='MarathonTime',data=running_data)
plt.show()


# The box-plot above of an athletes cross training (training other disciplines during their marathon training) provides us with little information as the datapoints are small. However, as we see new datapoints in the future, the model will pick up on the effect this additional training may have and could play a role in the improvement of the prediction model.

# In[86]:


from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
running_data['Marathon'] = label_encoder.fit_transform(running_data['Marathon']) + 1
running_data['Category'] = label_encoder.fit_transform(running_data['Category'])
running_data['CrossTraining'] = label_encoder.fit_transform(running_data['CrossTraining'])


# In[87]:


running_data


# The above data is cleaned using feature engineering to transform categorical variables to numerical variables. This will allow the data to form patterns in the features that correspond to a marathon timing.
# 
# Marathon was initialised with +1 as it is the first datapoint the model is viewing in terms of marathons. As the model starts to see new data, each new marathon viewed will be set to the consequent number of the viewed datapoints. ALthough the marathon plays no role in the current model, as the model begins to see new data, the marathon being run could play a role in the prediction.
# 
# As there are a finite number of categories it is easy to keep track of what numerical value corresponds to a category, this will allow us to correctly place new data in the right categories.
# 
# Although crosstraining could have several variations of datapoints, we will use a simplified method to track each datapoint by dividing them up by cycling, swimming. We can then assign the time spent on each activity to the different crosstraining disciplines and output numerical values for the model to pick up on.

# In[88]:


# Create target object and call it y
y = running_data['MarathonTime']

# Create X
features = ['Marathon', 'Category', 'km4week', 'sp4week', 'CrossTraining']
X = running_data[features]


# In[89]:


from sklearn.model_selection import train_test_split

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[90]:


from sklearn.tree import DecisionTreeRegressor

running_pred_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
running_pred_model.fit(train_X, train_y)


# In[91]:


val_predictions = running_pred_model.predict(val_X)


# In[92]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

val_mae = mean_absolute_error(val_y, val_predictions)
val_mae


# In[93]:


val_mape = mean_absolute_percentage_error(val_y, val_predictions)
val_mape


# In[94]:


from sklearn.ensemble import RandomForestRegressor

#Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

#fit the model
rf_model.fit(train_X, train_y)


# In[95]:


rf_val_predictions = rf_model.predict(val_X)


# In[96]:


# Calculate the mean absolute error of the Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)
rf_val_mae


# In[97]:


rf_val_mape = mean_absolute_percentage_error(rf_val_predictions, val_y)
rf_val_mape


# In[105]:


# giving inputs to the machine learning model
#features = ['Marathon', 'Category', 'km4week', 'sp4week', 'CrossTraining']
features = np.array([['1', '4', '1300.0', '14.029272', '5']])
# using inputs to predict the output
prediction = rf_model.predict(features)
print("Prediction: {}".format(prediction))


# https://www.linkedin.com/pulse/treating-removing-outliers-dataset-using-python-anubhav-tyagi/

# In[ ]:




