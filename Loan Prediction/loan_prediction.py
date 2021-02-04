import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('train_ctrUa4K.csv')
test_data = pd.read_csv('test_lAUu6dG.csv')
sample_submission = pd.read_csv('sample_submission_49d68Cx_input.csv',index_col=None)

label_encoder = LabelEncoder()

train_copy = train_data
test_copy = test_data

train_copy['Gender'].fillna(train_copy['Gender'].mode()[0], inplace=True)
test_copy['Gender'].fillna(test_copy['Gender'].mode()[0], inplace=True)

train_copy['Married'].fillna(train_copy['Married'].mode()[0], inplace=True)
train_copy['Dependents'].fillna(train_copy['Dependents'].mode()[0], inplace=True)
train_copy['Self_Employed'].fillna(train_copy['Self_Employed'].mode()[0], inplace=True)
train_copy['Credit_History'].fillna(train_copy['Credit_History'].mode()[0], inplace=True)


test_copy['Married'].fillna(test_copy['Married'].mode()[0], inplace=True)
test_copy['Dependents'].fillna(test_copy['Dependents'].mode()[0], inplace=True)
test_copy['Self_Employed'].fillna(test_copy['Self_Employed'].mode()[0], inplace=True)
test_copy['Credit_History'].fillna(test_copy['Credit_History'].mode()[0], inplace=True)

train_copy['Loan_Amount_Term'].fillna(train_copy['Loan_Amount_Term'].mode()[0], inplace=True)
test_copy['Loan_Amount_Term'].fillna(test_copy['Loan_Amount_Term'].mode()[0], inplace=True)

train_copy['LoanAmount'].fillna(train_copy['LoanAmount'].median(), inplace=True)
test_copy['LoanAmount'].fillna(test_copy['LoanAmount'].median(), inplace=True)

#To find sum of null values
null_sum_train = train_copy.isnull().sum()
null_sum_test = test_copy.isnull().sum()

train_copy['LoanAmount_log']=np.log(train_copy['LoanAmount'])
train_copy['LoanAmount'].hist(bins=20)
train_copy['LoanAmount_log'].hist(bins=20)
test_copy['LoanAmount_log']=np.log(test_copy['LoanAmount'])

#To drop 1st ID column
train_copy=train_copy.drop('Loan_ID',axis=1)
test_copy=test_copy.drop('Loan_ID',axis=1)


X = train_copy.drop('Loan_Status',1)
y = train_copy.Loan_Status

X = pd.get_dummies(X)
train_copy=pd.get_dummies(train_copy)
test_copy=pd.get_dummies(test_copy)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size=0.3)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression()

pred_cv = model.predict(x_cv)
score = accuracy_score(y_cv,pred_cv)
print(score)

pred_test = model.predict(test_copy)

sample_submission['Loan_Status']=pred_test
sample_submission['Loan_ID']=test_data['Loan_ID']

pd.DataFrame(sample_submission, columns=['Loan_ID','Loan_Status']).to_csv('sample_submission_49d68Cx.csv', index=False)


#new_submission['Loan_ID']= test_data['Loan_ID']
#new_submission['Loan_Status'] = pred_test











