import pandas as pd
data=pd.read_csv('heart.csv')
y=data['target']
x=data.drop('target',axis=1)

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=5)
clf.fit(xtrain,ytrain)
#Fitting model with trainig data

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
