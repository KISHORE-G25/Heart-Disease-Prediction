import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('heart.csv')    #to read the file
print(df.head())

# Create a plot to display the percentage of the positive and negative heart disease
labels = ['yes', 'No']
values = df['HeartDisease'].value_counts().values

plt.pie(values, labels=labels, autopct='%1.0f%%')
plt.title('HeartDisease')
plt.show()

# Display chest pain types based on the Heart Disease
pd.crosstab(df.ChestPainType,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
plt.title('Heart Disease Frequency According to Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(np.arange(4), ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'), rotation = 0)
plt.ylabel('Frequency')
plt.show()


# Get min, max and average of the age
print('Min age: ', min(df['Age']))
print('Max age: ', max(df['Age']))



# Display age distribution based on heart disease
sns.distplot(df[df['HeartDisease'] == 1]['Age'], label='Have heart disease')
sns.distplot(df[df['HeartDisease'] == 2]['Age'], label = 'Do not have heart disease')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution based on Heart Disease')
plt.show()


# Get min, max and average of the age of the people do not have heart diseas
print('Min age of people who do not have heart disease: ', min(df[df['HeartDisease'] == 1]['Age']))
print('Max age of people who do not have heart disease: ', max(df[df['HeartDisease'] == 1]['Age']))



le=LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])




NB = GaussianNB()

x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']      #to create the variable
print(x)
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)   #split the val
print(x_test)
print(y_test)


NB.fit(x_train, y_train)  #train the data

y_pred=NB.predict(x_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred))


##from sklearn.neighbors import KNeighborsClassifier
##knn = KNeighborsClassifier()
##knn.fit(x_train, y_train)
##y_pred_knn=knn.predict(x_test)
##print('KNN ACCURACY is', accuracy_score(y_test,y_pred_knn))





testPrediction = NB.predict([[36,1,1,120,166,0,1,138,0,0,2]])
if testPrediction==1:
    print(testPrediction,"The Patient Have Heart Disease,please consult the Doctor")
else:
    print(testPrediction,"The Patient is Normal")

import pickle
pickle.dump(NB,open('model.pkl','wb'))

##Sample Test
##39,1,2,120,339,0,1,170,0,0,2
##36,1,1,120,166,0,1,138,0,0,2
##51,0,0,120,0,1,1,127,1,1.5,2
##
##










