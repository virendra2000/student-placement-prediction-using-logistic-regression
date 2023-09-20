import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Placement.csv")

df.head()

df.info()

df.describe()

df.columns

df.shape

df['Placement'].value_counts()

df.groupby('Placement').mean()

y = df['Placement']

y.shape

y

X = df.drop(['Student_ID','Placement'],axis=1)

X.shape

X

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

X

X.shape

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=2529)

X_train.shape,X_test.shape,y_train.shape,y_test.shape

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

y_pred.shape

y_pred

lr.predict_proba(X_test)

y_test = np.array(y_test)
y_pred = np.array(y_pred)

# Bar graph banaate hain
x = np.arange(len(y_test))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, y_test, width, label='Actual (y_test)', color='blue')
plt.bar(x + width/2, y_pred, width, label='Predicted (y_pred)', color='green')

plt.xlabel('Data Points')
plt.ylabel('Placement')
plt.title('Actual vs. Predicted Placement')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

X_new = df.sample(1)

X_new

X_new.shape

X_new = X_new.drop(['Student_ID','Placement'],axis=1)

X_new

X_new.shape

X_new = ss.fit_transform(X_new)

y_pred_new = lr.predict(X_new)

y_pred_new

y_prob_new = lr.predict_proba(X_new)

print("Predicted Class:", y_pred_new)
print("Predicted Probabilities:", y_prob_new)

import matplotlib.pyplot as plt

# Assuming you have the data for y_pred and y_pred_new
y_pred_values = [y_pred[0], y_pred_new[0]]

# Create a histogram
plt.hist(y_pred, bins=10, color='blue', alpha=0.5, label='y_pred', edgecolor='black')
plt.hist(y_pred_new, bins=10, color='green', alpha=0.5, label='y_pred_new', edgecolor='black')

# Add labels and a title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of y_pred and y_pred_new')

# Add a legend
plt.legend()

# Show the plot
plt.show()