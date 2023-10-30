# Import Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from ml_from_scratch.neighbors import KNeighborsRegressor
from ml_from_scratch.neighbors import KNeighborsClassifier
import seaborn as sns

# read the dataset and filter out illogical data such as Glucose and BMI = 0
data = pd.read_csv('diabetes.csv')
data = data.drop(data[data["Glucose"] == 0].index).drop(data[data["BMI"] == 0].index)

# define Glucose and BMI as input and Outcome as output
X = data[["Glucose", "BMI"]]
y = data["Outcome"]

# split the train and test data with 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=123)


# define a function to scale the data since BMI and Glucose have a significantly different scale
def scaler_transform(data, scaler=StandardScaler()):
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled.columns = data.columns
    data_scaled.index = data.index
    return data_scaled


# scale both the train and test data
X_train_scaled = scaler_transform(data=X_train)
X_test_scaled = scaler_transform(data=X_test)

# fit the train data to the KNN algorithm
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(X_train_scaled, y_train)
# predict the outcome from the test data
y_pred = neigh.predict(X_test_scaled)

# evaluate and compare the accuracy and precision of various k value to choose the optimum value
accuracy_list = []
precision_list = []
k_val = np.arange(5, 55, 5)
for k in k_val:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train_scaled, y_train)

    y_pred = neigh.predict(X_test_scaled)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))

# lineplot to visualize the accuracy and precision of various k value
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 5))

ax1 = sns.lineplot(x=k_val, y=accuracy_list, ax=ax[0])
ax1.set_xlabel("K value", fontsize=12)
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.axvline(x=30, color='red', linestyle='--')
ax2 = sns.lineplot(x=k_val, y=precision_list, ax=ax[1])
ax2.set_xlabel("K value", fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.axvline(x=30, color='red', linestyle='--')

fig.suptitle('Optimal K value Selection', fontsize=18)

# # scatterplot of the whole dataset
# ax = sns.scatterplot(data=data, x=data["Glucose"], y=data["BMI"], hue=data["Outcome"])
# ax.set_xlabel("Glucose", fontsize=12)
# ax.set_ylabel("BMI", fontsize=12)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ['No Diabetes', 'Diabetes'], loc='upper right')

plt.show()
