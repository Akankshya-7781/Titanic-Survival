# Titanic-Survival
This contains an analysis of the survival rate of passengers of the Titanic Shipwreck
#Importing of datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


#Data Acquiring

train_data=pd.read_csv('C:\\Users\\USER\\Downloads\\train.csv')
test_data=pd.read_csv('C:\\Users\\USER\\Downloads\\test.csv')


#Data Modification

train_df=pd.DataFrame(train_data)
test_df=pd.DataFrame(test_data)

train_df
test_df.head()

#DataCleaning
train_df.dropna()
test_df.dropna()


#Data Analysis

train_df.describe()

#Checking the survival rate in terms of individual characteristics.
#That is, we will be checking here how the rate of survival depends on the gender,Pclass,,fare and so on.

#By Sex column.
gender_survival=train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
gender_survival

#By Passenger Class
Pclass_survival=train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
Pclass_survival

#By Age
Age_survival=train_df[['Age','Survived']].groupby(['Age'],as_index=False).mean().sort_values(by='Survived',ascending=False)
Age_survival

#By SibSp
SibSp_survival=train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)
SibSp_survival

#By Parch
Parch_survival=train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)
Parch_survival

#By Fare
Fare_survival=train_df[['Fare','Survived']].groupby(['Fare'],as_index=False).mean().sort_values(by='Survived',ascending=False)
Fare_survival

# By the port of embarkation
Embarked_survival=train_df[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
Embarked_survival


#Visualising the Analysis

#Plot of survival rate based on gender
plt.plot(gender_survival['Sex'].values,gender_survival['Survived'].values)
plt.ylabel('Survived')
plt.xlabel('Sex')
plt.title('Survival rate based on gender of passenger')

#Plot of survival rate of passengers based on class of travelling
plt.plot(Pclass_survival['Pclass'].values,Pclass_survival['Survived'].values)
plt.ylabel('Survived')
plt.xlabel('Pclass')
plt.title('Survival rate based on traveling class of passenger')

#Plot of survival rate based on age
plt.plot(Age_survival['Survived'].values,Age_survival['Age'].values)
plt.xlabel('Survived')
plt.ylabel('Age')
plt.title('Survival rate based on age of passenger')

#Plot of survival rate based on fare given by passenger
plt.plot(Fare_survival['Survived'].values,Fare_survival['Fare'].values)
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.title('Survival rate based on fare given by passenger')

#Plot of survival rate based on parch of passengers
plt.plot(Parch_survival['Survived'].values,Parch_survival['Parch'].values)
plt.xlabel('Survived')
plt.ylabel('Parch')
plt.title('Survival rate based on parch of passenger')

#plot of survival rate based on number of siblings and spouses of passenger
plt.plot(SibSp_survival['Survived'].values,SibSp_survival['SibSp'].values)
plt.xlabel('Survived')
plt.ylabel('SibSp')
plt.title('Survival rate based on number of siblings and spouses of passenger')

plt.plot(Embarked_survival['Embarked'].values,Embarked_survival['Survived'].values)
plt.ylabel('Survived')
plt.xlabel('Embarked')
plt.title('Survival rate based on the port of embarktion of passenger')


#Deep Analysis


#Binning the passengers by their ages that is dividing them into age groups
Age_survival['Age'].min()
Age_survival['Age'].max()
s1=0
s2=0
s3=0
s4=0
s5=0
s6=0
s7=0
s8=0
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0
c7=0
c8=0
#Iterating through the rows to calculate the net survival of each age-group and total number of passenger in each group
for i,row in Age_survival.iterrows():
    if row['Age']>0 and row['Age']<=10:
        s1=s1+row['Survived']
        c1=c1+1
    elif row['Age']>10 and row['Age']<=20:
        s2=s2+row['Survived']
        c2=c2+1
    elif row['Age']>20 and row['Age']<=30:
        s3=s3+row['Survived']
        c3=c3+1
    elif row['Age']>30 and row['Age']<=40:
        s4=s4+row['Survived']
        c4=c4+1
    elif row['Age']>40 and row['Age']<=50:
        s5=s5+row['Survived']
        c5=c5+1
    elif row['Age']>50 and row['Age']<=60:
        s6=s6+row['Survived']
        c6=c6+1
    elif row['Age']>60 and row['Age']<=70:
        s7=s7+row['Survived']
        c7=c7+1
    else:
        s8=s8+row['Survived']
        c8=+1
#Calculating the mean survival rate of each age-group
S1=s1/c1
S2=s2/c2
S3=s3/c3
S4=s4/c4
S5=s5/c5
S6=s6/c6
S7=s7/c7
S8=s8/c8

#Creating a new dataframe of survival rate of passengers based on the Age-Group
ag_survival=[['0-10',S1],['10-20',S2],['20-30',S3],['30-40',S4],['40-50',S5],['50-60',S6],['60-70',S7],['70-80',S8]]
AG_survival=pd.DataFrame(ag_survival,columns=['Age-Group','Mean Survival Rate'])
AG_survival

#plot of mean survival rate based on age group of passenger
plt.plot(AG_survival['Age-Group'].values,AG_survival['Mean Survival Rate'])
plt.xlabel('Age-Groups')
plt.ylabel('Mean Survival Rate')
plt.title('Mean Survival rate based on age-group in which the passenger lies')


#Diving the passengers into groups based on the fare of tickets
Fare_survival['Fare'].min()
Fare_survival['Fare'].max()
s1=0
s2=0
s3=0
s4=0
s5=0
s6=0
s7=0
s8=0
s9=0
s10=0
s11=0
c1=0
c2=0
c3=0
c4=0
c5=0
c6=0
c7=0
c8=0
c9=0
c10=0
c11=0
#Iterating through the rows to calculate the net survival of each Fare-Range and total number of passenger in each range
for i,row in Fare_survival.iterrows():
    if row['Fare']>0.0 and row['Fare']<=50.0:
        s1=s1+row['Survived']
        c1=c1+1
    elif row['Fare']>50.0 and row['Fare']<=100.0:
        s2=s2+row['Survived']
        c2=c2+1
    elif row['Fare']>100.0 and row['Fare']<=150.0:
        s3=s3+row['Survived']
        c3=c3+1
    elif row['Fare']>150.0 and row['Fare']<=200.0:
        s4=s4+row['Survived']
        c4=c4+1
    elif row['Fare']>200.0 and row['Fare']<=250.0:
        s5=s5+row['Survived']
        c5=c5+1
    elif row['Fare']>250.0 and row['Fare']<=300.0:
        s6=s6+row['Survived']
        c6=c6+1
    elif row['Fare']>300.0 and row['Fare']<=350.0:
        s7=s7+row['Survived']
        c7=c7+1
    elif row['Fare']>350.0 and row['Fare']<=400.0:
        s8=s8+row['Survived']
        c8=+1
    elif row['Fare']>400.0 and row['Fare']<=450.0:
        s9=s9+row['Survived']
        c9=c9+1
    elif row['Fare']>450.0 and row['Fare']<=500.0:
        s10=s10+1
        c10=c10+1
    else:
        s11=s11+1
        c11=c11+1
    
#Calculating the mean survival rate of each age-group
S1=s1/c1
S2=s2/c2
S3=s3/c3
S4=s4/c4
S5=s5/c5
S6=s6/c6
S11=s11/c11

S7=0
S8=0
S9=0
S10=0

#Creating a new dataframe of survival rate of passengers based on the Age-Group
fare_survival=[['0-50.0',S1],['50.0-100.0',S2],['100.0-150.0',S3],['150.0-200.0',S4],['200.0-250.0',S5],['250.0-300.0',S6],['300.0-350.0',S7],['350.0-400.0',S8],['400.0-450.0',S9],['450.0-500.0',S10],['500.0-550.0',S11]]
Fare_survival=pd.DataFrame(fare_survival,columns=['Fare-range','Mean Survival Rate'])
Fare_survival
#In the above section I have considered S7 to be zero because there are no passengers in that range of fare.Thus no passenger on board had paid ticket fare in that range so both s7 andZ c7 are zero.Hence,it throws an error of ZerodividedbyZero. And in a similar for S8,S9 and S10.

#plot of mean survival rate based on Fare-range of passenger
plt.plot(Fare_survival['Fare-range'].values,Fare_survival['Mean Survival Rate'])
plt.xlabel('Fare-range')
plt.ylabel('Mean Survival Rate')
plt.title('Mean Survival rate based on Fare-range in which the passenger lies')


#Combining the survival rates based on age and sex of the passenger

#Initialising separate variables for calculating the mean survival rate of females of a particular age group from the train_df dataframe
sf1=0
sf2=0
sf3=0
sf4=0
sf5=0
sf6=0
sf7=0
sf8=0
cf1=0
cf2=0
cf3=0
cf4=0
cf5=0
cf6=0
cf7=0
cf8=0

#Initialising separate variables for calculating the mean survival rate of females of a particular age group from the train_df dataframe
sm1=0
sm2=0
sm3=0
sm4=0
sm5=0
sm6=0
sm7=0
sm8=0
cm1=0
cm2=0
cm3=0
cm4=0
cm5=0
cm6=0
cm7=0
cm8=0

#Iterating through the rows
for i,row in train_df.iterrows():
    if row['Age']>0 and row['Age']<=10:
            if row['Sex']=='female':
                sf1=sf1+row['Survived']
                cf1=cf1+1
            if row['Sex']=='male':
                sm1=sm1+row['Survived']
                sm1=sm1+1
    
    if row['Age']>10 and row['Age']<=20:
            if row['Sex']=='female':
                sf2=sf2+row['Survived']
                cf2=cf2+1
            if row['Sex']=='male':
                sm2=sm2+row['Survived']
                sm2=sm2+1
                
    if row['Age']>20 and row['Age']<=30:
            if row['Sex']=='female':
                sf3=sf3+row['Survived']
                cf3=cf3+1
            if row['Sex']=='male':
                sm3=sm3+row['Survived']
                sm3=sm3+1
                
    if row['Age']>30 and row['Age']<=40:
            if row['Sex']=='female':
                sf4=sf4+row['Survived']
                cf4=cf4+1
            if row['Sex']=='male':
                sm4=sm4+row['Survived']
                sm4=sm4+1
                
    if row['Age']>40 and row['Age']<=50:
            if row['Sex']=='female':
                sf5=sf5+row['Survived']
                cf5=cf5+1
            if row['Sex']=='male':
                sm5=sm5+row['Survived']
                sm5=sm5+1
                
    if row['Age']>50 and row['Age']<=60:
            if row['Sex']=='female':
                sf6=sf6+row['Survived']
                cf6=cf6+1
            if row['Sex']=='male':
                sm6=sm6+row['Survived']
                sm6=sm6+1
    if row['Age']>60 and row['Age']<=70:
            if row['Sex']=='female':
                sf7=sf7+row['Survived']
                cf7=cf7+1
            if row['Sex']=='male':
                sm7=sm7+row['Survived']
                sm7=sm7+1
                
    if row['Age']>70 and row['Age']<=80:
            if row['Sex']=='female':
                sf8=sf8+row['Survived']
                cf8=cf8+1
            if row['Sex']=='male':
                sm8=sm8+row['Survived']
                sm8=sm8+1

#Calculating the mean survival rate of females of a particular age group
Sf1=sf1/cf1
Sf2=sf2/cf2
Sf3=sf3/cf3
Sf4=sf4/cf4
Sf5=sf5/cf5
Sf6=sf6/cf6
Sf7=sf7/cf7
Sf8=0#I have assigned this as 0 because in this case both sf8 and cf8 are zero

#Calculating the mean survival rate of males of a particular age group
Sm1=0#I have assigned this as 0 because in this case both sm1 and cm1 are zero
Sm2=0#I have assigned this as 0 because in this case both sm2 and cm2 are zero
Sm3=0#I have assigned this as 0 because in this case both sm3 and cm3 are zero
Sm4=0#I have assigned this as 0 because in this case both sm4 and cm4 are zero
Sm5=0#I have assigned this as 0 because in this case both sm5 and cm5 are zero
Sm6=0#I have assigned this as 0 because in this case both sm6 and cm6 are zero 
Sm7=0#I have assigned this as 0 because in this case both sm7 and cm7 are zero
Sm8=0#I have assigned this as 0 because in this case both sm8 and cm8 are zero

#Creating dataframe of survival rate of male and female passengers based on their age group
ags_survival=[['0-10',Sf1,Sm1],['10-20',Sf2,Sm2],['20-30',Sf3,Sm3],['30-40',Sf4,Sm4],['40-50',Sf5,Sm5],['50-60',Sf6,Sm6],['60-70',Sf7,Sm7],['70-80',Sf8,Sm8]]
AGs_survival=pd.DataFrame(ags_survival,columns=['Age-Group','Mean Survival Rate of Females','Mean Survival Rate of Males'])
AGs_survival



#Combining the survival rate based on the fare of ticket and sex of each passenger

#Initialising separate variables for calculating the mean survival rate of females of a particular passenger class from the train_df dataframe
spf1=0
spf2=0
spf3=0
cpf1=0
cpf2=0
cpf3=0


#Initialising separate variables for calculating the mean survival rate of females of a particular passenger class from the train_df dataframe
spm1=0
spm2=0
spm3=0
cpm1=0
cpm2=0
cpm3=0

#Iterating through the rows
for i,row in train_df.iterrows():
    if row['Pclass']==1:

            if row['Sex']=='female':
                spf1=spf1+row['Survived']
                cpf1=cpf1+1
            else:
                spm1=spm1+row['Survived']
                spm1=spm1+1
    
    if row['Pclass']==2:
            if row['Sex']=='female':
                spf2=spf2+row['Survived']
                cpf2=cpf2+1
            else:
                spm2=spm2+row['Survived']
                spm2=spm2+1
                
    if row['Pclass']==3:
            if row['Sex']=='female':
                spf3=spf3+row['Survived']
                cpf3=cpf3+1
            else:
                spm3=spm3+row['Survived']
                spm3=spm3+1
                
#Calculating the mean survival rate of females and males of a particular age group
Spf1=spf1/cpf1
Spf2=spf2/cpf2
Spf3=spf3/cpf3
Spm3=0#I have considered it to be 0 because both spm3 and cpm3 are 0.
Spm2=0#I have considered it to be 0 because both spm2 and cpm2 are 0.
Spm1=0#I have considered it to be 0 because both spm1 and cpm1 are 0.

#Creating a new data frame for the survival rate of male anf female passengers based on the passenger class they travelled in.
spf_survival=[[1,Spf1,Spm1],[2,Spf2,Spm2],[3,Spf3,Spm3]]
SPF_survival=pd.DataFrame(spf_survival,columns=['Passenger class','Mean survival rate of females','Mean survival rate of males'])
SPF_survival


#Combining the survival rate of based on the age group and parch of the passenger

#Initialising separate variables for calculating the mean survival rate of passengers of different age groups travelling in  different passenger class from the train_df dataframe
sp11=0
sp12=0
sp13=0
sp21=0
sp22=0
sp23=0
sp31=0
sp32=0
sp33=0
sp41=0
sp42=0
sp43=0
sp51=0
sp52=0
sp53=0
sp61=0
sp62=0
sp63=0
sp71=0
sp72=0
sp73=0
sp81=0
sp82=0
sp83=0
cp11=0
cp12=0
cp13=0
cp21=0
cp22=0
cp23=0
cp31=0
cp32=0
cp33=0
cp41=0
cp42=0
cp43=0
cp51=0
cp52=0
cp53=0
cp61=0
cp62=0
cp63=0
cp71=0
cp72=0
cp73=0
cp81=0
cp82=0
cp83=0

#Iterating through the rows
for i,row in train_df.iterrows():
    if row['Age']>0 and row['Age']<=10:
            if row['Pclass']==1 :
                sp11=sp11+row['Survived']
                cp11=cp11+1
            if row['Pclass']==2:
                sp12=sp12+row['Survived']
                cp12=cp12+1
            if row['Pclass']==3:
                sp13=sp13+row['Survived']
                cp13=cp13+1
    if row['Age']>10 and row['Age']<=20:
            if row['Pclass']==1 :
                sp21=sp21+row['Survived']
                cp21=cp21+1
            if row['Pclass']==2:
                sp22=sp22+row['Survived']
                cp22=cp22+1
            if row['Pclass']==3:
                sp23=sp23+row['Survived']
                cp23=cp23+1
                
    if row['Age']>20 and row['Age']<=30:
            if row['Pclass']==1 :
                sp31=sp31+row['Survived']
                cp31=cp31+1
            if row['Pclass']==2:
                sp32=sp32+row['Survived']
                cp32=cp32+1
            if row['Pclass']==3:
                sp33=sp33+row['Survived']
                cp33=cp33+1
    if row['Age']>30 and row['Age']<=40:
            if row['Pclass']==1 :
                sp41=sp41+row['Survived']
                cp41=cp41+1
            if row['Pclass']==2:
                sp42=sp42+row['Survived']
                cp42=cp42+1
            if row['Pclass']==3:
                sp43=sp43+row['Survived']
                cp43=cp43+1
                
    if row['Age']>40 and row['Age']<=50:
            if row['Pclass']==1 :
                sp51=sp51+row['Survived']
                cp51=cp51+1
            if row['Pclass']==2:
                sp52=sp52+row['Survived']
                cp52=cp52+1
            if row['Pclass']==3:
                sp13=sp13+row['Survived']
                cp13=cp13+1
    if row['Age']>50 and row['Age']<=60:
            if row['Pclass']==1 :
                sp61=sp61+row['Survived']
                cp61=cp61+1
            if row['Pclass']==2:
                sp62=sp62+row['Survived']
                cp62=cp62+1
            if row['Pclass']==3:
                sp63=sp63+row['Survived']
                cp63=cp63+1
    if row['Age']>60 and row['Age']<=70:
            if row['Pclass']==1 :
                sp71=sp71+row['Survived']
                cp71=cp71+1
            if row['Pclass']==2:
                sp72=sp72+row['Survived']
                cp72=cp72+1
            if row['Pclass']==3:
                sp73=sp73+row['Survived']
                cp73=cp73+1
                
    if row['Age']>70 and row['Age']<=80:
            if row['Pclass']==1 :
                sp81=sp81+row['Survived']
                cp81=cp81+1
            if row['Pclass']==2:
                sp82=sp82+row['Survived']
                cp82=cp82+1
            if row['Pclass']==3:
                sp83=sp83+row['Survived']
                cp83=cp83+1

#Calculating the mean survival rate of passengers of different passenger class of a particular age group
Sp11=sp11/cp11
Sp12=sp12/cp12
Sp13=sp13/cp13
Sp21=sp21/cp21
Sp22=sp22/cp22
Sp23=sp23/cp23
Sp31=sp31/cp31
Sp32=sp32/cp32
Sp33=sp33/cp33
Sp41=sp41/cp41
Sp42=sp42/cp42
Sp43=sp43/cp43
Sp51=sp51/cp51
Sp52=sp52/cp52
Sp53=0#I have assigned this as 0 because in this case both sp53 and cp53 are zero
Sp61=sp61/cp61
Sp62=sp62/cp62
Sp63=sp63/cp63
Sp71=sp71/cp71
Sp72=sp72/cp72
Sp73=sp73/cp73
Sp81=sp81/cp81
Sp82=0#I have assigned this as 0 because in this case both sp82 and cp82 are zero
Sp83=sp83/cp83
#Creating dataframe of survival rate of male and female passengers based on their age group

sp_survival=[['0-10',Sp11,Sp12,Sp13],['10-20',Sp21,Sp22,Sp23],['20-30',Sp31,Sp32,Sp33],['30-40',Sp41,Sp42,Sp43],['40-50',Sp51,Sp52,Sp53],['50-60',Sp61,Sp62,Sp63],['60-70',Sp71,Sp72,Sp73],['70-80',Sp81,Sp82,Sp83]]
SP_survival=pd.DataFrame(sp_survival,columns=['Age-Group','Mean Survival Rate of Pclass1','Mean Survival Rate of Pclass2','Mean Srvival Rate of Pclass3'])
SP_survival


#Deleting the rows not required
del train_df['Name']
del train_df['Ticket']
del train_df['Cabin']
del train_df['SibSp']
del test_df['Name']
del test_df['Ticket']
del test_df['Cabin']
del test_df['SibSp']


train_df=train_df.fillna(method='ffill')
train_df
test_df=test_df.fillna(method='ffill')
test_df

#Converting the string notifications or data to digits
train_df=train_df.replace(to_replace=['S','C','Q'],value=[0,1,2])
train_df
test_df=test_df.replace(to_replace=['S','C','Q'],value=[0,1,2])
test_df

#Converting the string notifications or data to digits
train_df=train_df.replace(to_replace=['male','female'],value=[0,1])
train_df
test_df=test_df.replace(to_replace=['male','female'],value=[0,1])
test_df

necessary_columns=['Pclass','Sex','Parch','Age','Fare','Embarked']
X_train=train_df[necessary_columns]
Y_train=train_df['Survived']
X_test=test_df[necessary_columns]
X_test.head()


#Prediction and Measurement of Accuracy

X_train.shape,Y_train.shape,X_test.shape
X_train.head()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

#MultinomialNb
MultiNb=MultinomialNB()
MultiNb.fit(X_train,Y_train)
y_pred=MultiNb.predict(X_test)
acc_multinomial=round(MultiNb.score(X_train,Y_train)*100,2)
acc_multinomial

#KNearestNeighbor
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)
acc_KNN = round(KNN.score(X_train, Y_train) * 100, 2)
acc_KNN

# Gaussian Naive Bayes
GaussNb = GaussianNB()
GaussNb.fit(X_train, Y_train)
Y_pred = GaussNb.predict(X_test)
acc_gaussian = round(GaussNb.score(X_train, Y_train) * 100, 2)
acc_gaussian



#Final Presentation

final_pred=pd.DataFrame({'MODEL':['LogisticRegression',
 'DecisionTreeClassifier','MultinomialNB','GaussianNB','KNearestNeighbour'],'SCORES':[acc_log,acc_decision_tree,acc_multinomial,acc_KNN,acc_gaussian]})
                                                                                                                                                                          
final_pred.sort_values(by='SCORES',ascending=True)
