import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('CustomersData.csv')
print(data.head())
print()

# scatterPlot of Age vs Spending Score (1-100)
plt.scatter(data.Age, data['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100)')
plt.show()

# ScatterPlot of Annual Income (k$) vs Spending Score (1-100)
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100)')
plt.show()

#  fitting the data first to 0-1 range so there is no disparity in ranges
duplicate = data
scaler = MinMaxScaler()
scaler.fit(duplicate[['Age']])
duplicate.Age = scaler.transform(duplicate[['Age']])

scaler.fit(duplicate[['Annual Income (k$)']])
duplicate['Annual Income (k$)'] = scaler.transform(duplicate[['Annual Income (k$)']])

scaler.fit(duplicate[['Spending Score (1-100)']])
duplicate['Spending Score (1-100)'] = scaler.transform(duplicate[['Spending Score (1-100)']])


# K-means with k=3 for Age and Spending

km = KMeans(n_clusters=3, n_init=10)
features = duplicate[['Age', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())


data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]


plt.scatter(data1.Age, data1['Spending Score (1-100)'], color='red')
plt.scatter(data2.Age, data2['Spending Score (1-100)'], color='green')
plt.scatter(data3.Age, data3['Spending Score (1-100)'], color='blue')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100) using k = 3')
plt.show()

# K-means with k=4 for Age and Spending

km = KMeans(n_clusters=4, n_init=10)
features = duplicate[['Age', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]
data4 = duplicate[duplicate.Cluster == 3]

plt.scatter(data1.Age, data1['Spending Score (1-100)'], color='red')
plt.scatter(data2.Age, data2['Spending Score (1-100)'], color='green')
plt.scatter(data3.Age, data3['Spending Score (1-100)'], color='blue')
plt.scatter(data4.Age, data4['Spending Score (1-100)'], color='orange')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100) using k=4')
plt.show()

# K-means with k=5 for Age and Spending

km = KMeans(n_clusters=5, n_init=10)
features = duplicate[['Age', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]
data4 = duplicate[duplicate.Cluster == 3]
data5 = duplicate[duplicate.Cluster == 4]

plt.scatter(data1.Age, data1['Spending Score (1-100)'], color='red')
plt.scatter(data2.Age, data2['Spending Score (1-100)'], color='green')
plt.scatter(data3.Age, data3['Spending Score (1-100)'], color='blue')
plt.scatter(data4.Age, data4['Spending Score (1-100)'], color='orange')
plt.scatter(data5.Age, data5['Spending Score (1-100)'], color='purple')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100) using k=5')
plt.show()

# K-means with k=3 for Annual Income and Spending

km = KMeans(n_clusters=3, n_init=10)
features = duplicate[['Annual Income (k$)', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]

plt.scatter(data1['Annual Income (k$)'], data1['Spending Score (1-100)'], color='red')
plt.scatter(data2['Annual Income (k$)'], data2['Spending Score (1-100)'], color='green')
plt.scatter(data3['Annual Income (k$)'], data3['Spending Score (1-100)'], color='blue')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k = 3')
plt.show()


# K-means with k=4 for Annual income and Spending

km = KMeans(n_clusters=4, n_init=10)
features = duplicate[['Annual Income (k$)', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]
data4 = duplicate[duplicate.Cluster == 3]

plt.scatter(data1['Annual Income (k$)'], data1['Spending Score (1-100)'], color='red')
plt.scatter(data2['Annual Income (k$)'], data2['Spending Score (1-100)'], color='green')
plt.scatter(data3['Annual Income (k$)'], data3['Spending Score (1-100)'], color='blue')
plt.scatter(data4['Annual Income (k$)'], data4['Spending Score (1-100)'], color='orange')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k = 4')
plt.show()

# K-means with k=5 for Annual income and Spending

km = KMeans(n_clusters=5, n_init=10)
features = duplicate[['Annual Income (k$)', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

data1 = duplicate[duplicate.Cluster == 0]
data2 = duplicate[duplicate.Cluster == 1]
data3 = duplicate[duplicate.Cluster == 2]
data4 = duplicate[duplicate.Cluster == 3]
data5 = duplicate[duplicate.Cluster == 4]

plt.scatter(data1['Annual Income (k$)'], data1['Spending Score (1-100)'], color='red')
plt.scatter(data2['Annual Income (k$)'], data2['Spending Score (1-100)'], color='green')
plt.scatter(data3['Annual Income (k$)'], data3['Spending Score (1-100)'], color='blue')
plt.scatter(data4['Annual Income (k$)'], data4['Spending Score (1-100)'], color='orange')
plt.scatter(data5['Annual Income (k$)'], data5['Spending Score (1-100)'], color='brown')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k = 5')
plt.show()


#  3-d Clustering

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
k_values = [2, 3, 4, 5]
for k in k_values:
    # Perform K-means clustering
    km = KMeans(n_clusters=k,n_init=10)
    km.fit(features)
    cluster_labels = km.labels_

    # Visualize the clusters in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features['Age'], features['Annual Income (k$)'], features['Spending Score (1-100)'], c=cluster_labels)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income (k$)')
    ax.set_zlabel('Spending Score (1-100)')
    ax.set_title(f'3D Clustering (K={k})')

    plt.show()
