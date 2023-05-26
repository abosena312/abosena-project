# we will talk about Apple stock data
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

apple_data = pd.read_csv('AAPL.csv')
print('--------------------------data_test---------------------')
print(apple_data)
print(apple_data.shape)
print('---------data.coulmns----------')
print(apple_data.head(0))
print('---------data.describ----------')
print(apple_data.describe())
print('---------data.duplicated----------')
print(apple_data.duplicated().sum())
print('---------empty cell----------')
print(apple_data.isnull().sum())
print('--------------------------data_train---------------------')
# to draw the data

sns.heatmap(apple_data.corr(), annot=True)
plt.show()


plt.xlabel('Open')
plt.scatter(apple_data['Open'], apple_data['Close'])
plt.ylabel('Close')
plt.show()
x = apple_data.iloc[:, :-1].values
y = apple_data.iloc[:, -1].values
scaler = preprocessing.MinMaxScaler()
scaled_data_train = scaler.fit_transform(x)
scaled_data_train = pd.DataFrame(scaled_data_train, columns=apple_data.columns[:-1])
print(f'the data _scaled= \n {scaled_data_train}')
# we imported the train test split method to train the module to the data

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
print('------------------data before val--------------------')
print(f'X_train =\n\n{X_train}')
print(f'X_test=\n\n {X_test}')
print(f'Y_train=\n \n{Y_train}')
print(f'Y_test= \n\n{Y_test}')


# we used the logistic regression 
lg = LogisticRegression()

lg.fit(X_train, Y_train)
Y_pred = lg.predict(X_test)
co = confusion_matrix(Y_test, Y_pred)
print('confusion_matrix to LogisticRegression=\n', co)
sns.heatmap(co, annot=True)
plt.show()
re = recall_score(Y_test, Y_pred, average='micro')
print('recall_score to LogisticRegression', re)
f1 = f1_score(Y_test, Y_pred, average='micro')
print('f1_score  to LogisticRegression=', f1)


clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)
Y_pred2 = clf.predict(X_test)
co_2 = confusion_matrix(Y_test, Y_pred)
print('confusion_matrix to SVC=\n', co)
sns.heatmap(co_2, annot=True)
plt.show()
re2 = recall_score(Y_test, Y_pred, average='micro')
print('recall_score to svc', re2)
f1_2 = f1_score(Y_test, Y_pred, average='micro')
print('f1_score  to svc=', f1_2)