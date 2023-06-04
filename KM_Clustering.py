import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('CustomersData.csv')
print(data.head())
print()

# scatterPlot of Age vs Spending Score (1-100)
sns.scatterplot(data=data, x='Age', y='Spending Score (1-100)')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100)')
plt.show()

# ScatterPlot of Annual Income (k$) vs Spending Score (1-100)
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100)')
plt.show()

# fitting the data first to 0-1 range so there is no disparity in ranges
duplicate = data.copy()
scaler = MinMaxScaler()
duplicate[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] = scaler.fit_transform(duplicate[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# K-means with k=3 for Age and Spending
km = KMeans(n_clusters=3, n_init=10)
features = duplicate[['Age', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

# ScatterPlot of Age vs Spending Score (1-100) using k=3
sns.scatterplot(data=duplicate, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
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

# ScatterPlot of Age vs Spending Score (1-100) using k=4
sns.scatterplot(data=duplicate, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100) using k = 4')
plt.show()

# K-means with k=5 for Age and Spending
km = KMeans(n_clusters=5, n_init=10)
features = duplicate[['Age', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

# ScatterPlot of Age vs Spending Score (1-100) using k=5
sns.scatterplot(data=duplicate, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Age vs Spending Score (1-100) using k = 5')
plt.show()

# K-means with k=3 for Annual Income and Spending
km = KMeans(n_clusters=3, n_init=10)
features = duplicate[['Annual Income (k$)', 'Spending Score (1-100)']]
y_predicted = km.fit_predict(features)
duplicate['Cluster'] = y_predicted
print(duplicate.head())

# ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k=3
sns.scatterplot(data=duplicate, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
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

# ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k=4
sns.scatterplot(data=duplicate, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
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

# ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k=5
sns.scatterplot(data=duplicate, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('ScatterPlot of Annual Income (k$) vs Spending Score (1-100) using k = 5')
plt.show()

# 3D Clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
k_values = [2, 3, 4, 5]
for k in k_values:
    # Perform K-means clustering
    km = KMeans(n_clusters=k, n_init=10)
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
