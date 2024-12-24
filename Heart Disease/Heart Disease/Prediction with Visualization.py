import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('heart.csv')    #to read the file
print(df.head())


### Create a plot to display the percentage of the positive and negative heart disease
##labels = ['yes', 'No']
##values = df['HeartDisease'].value_counts().values
##
##plt.pie(values, labels=labels, autopct='%1.0f%%')
##plt.title('HeartDisease')
##plt.show()
##
##
### Display chest pain types based on the Heart Disease
##pd.crosstab(df.ChestPainType,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
##plt.title('Heart Disease Frequency According to Chest Pain Type')
##plt.xlabel('Chest Pain Type')
##plt.xticks(np.arange(4), ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'), rotation = 0)
##plt.ylabel('Frequency')
##plt.show()
##
##
### Get min, max and average of the age
##print('Min age: ', min(df['Age']))
##print('Max age: ', max(df['Age']))
##
##
### Display Heart Disease  According to Gender
##pd.crosstab(df.Sex,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
##plt.title('Heart Disease  According to Gender')
##plt.xlabel('male & Female')
##plt.xticks(np.arange(2), ('male', 'female'), rotation = 0)
##plt.ylabel('Count')
##plt.show()
##
##
### Display Heart Disease  According to RestingECG
##pd.crosstab(df.RestingECG,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
##plt.title('Heart Disease  According to RestingECG')
##plt.xlabel('RestingECG')
##plt.xticks(np.arange(3), ('Left ventricular hypertrophy', ' ST Elevation Myocardial Infarction.', 'Normal'), rotation = 0)
##plt.ylabel('Count')
##plt.show()
##
### Display Heart Disease  According to ExerciseAngina
##pd.crosstab(df.ExerciseAngina,df.HeartDisease).plot(kind = "bar", figsize = (8, 6))
##plt.title('Heart Disease  According to ExerciseAngina')
##plt.xlabel('ExerciseAngina')
##plt.xticks(np.arange(2), ( 'No','Yes'), rotation = 0)
##plt.ylabel('Count')
##plt.show()
##
##
### Display age distribution based on heart disease
##sns.distplot(df[df['HeartDisease'] == 1]['Age'], label='Have heart disease')
##sns.distplot(df[df['HeartDisease'] == 2]['Age'], label = 'Do not have heart disease')
##plt.xlabel('Age')
##plt.ylabel('Frequency')
##plt.title('Age Distribution based on Heart Disease')
##plt.show()
##
##
##
##
### Get min, max and average of the age of the people do not have heart diseas
##print('Min age of people who do not have heart disease: ', min(df[df['HeartDisease'] == 1]['Age']))
##print('Max age of people who do not have heart disease: ', max(df[df['HeartDisease'] == 1]['Age']))
##
##
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])

print(df.head())


x=df.drop(columns=['HeartDisease'])
y=df['HeartDisease']      #to create the variable
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)   #split the val
print(x_test)
print(y_test)



from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)  #train the data


y_pred=NB.predict(x_test)
##print(y_pred)
##print(y_test)
##
from sklearn.metrics import accuracy_score
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred))

##


testPrediction = NB.predict([[19,1,4,120,166,0,1,138,0,0,2]])
if testPrediction==1:
    print("The Patient Have Heart Disease,please consult the Doctor")
else:
    print("The Patient Normal")







##from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score ,confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)






