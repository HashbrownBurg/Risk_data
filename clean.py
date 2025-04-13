import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

credit = pd.read_csv('credit_risk_dataset.csv')

credit.dropna(inplace=True)

home_ownership_map = {
    "RENT": 1,
    "MORTGAGE": 2,
    "OWN": 3
}
credit['person_home_ownership'] = credit['person_home_ownership'].map(home_ownership_map)

credit = credit.loc[credit['person_income'] < 500000]

credit.dropna(inplace=True)

credit['cb_person_default_on_file'] = credit['cb_person_default_on_file'].map({'N': 0, 'Y': 1})

loan_intent_map = {
    "PERSONAL": 1,
    "EDUCATION": 2,
    "MEDICAL": 3,
    "VENTURE": 4,
    "HOMEIMPROVEMENT": 5,
    "DEBTCONSOLIDATION": 6
}

credit['loan_intent'] = credit['loan_intent'].map(loan_intent_map)

# Save cleaned data
credit.to_csv('clean_risk.csv', index=False)
credit = pd.read_csv('clean_risk.csv')

credit_numeric = credit.select_dtypes(include='number')

inertias = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=23)
    km.fit(credit_numeric)
    inertias.append(km.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=4, random_state=345)
kmeans.fit(credit_numeric)
labels = kmeans.labels_

print(pd.Series(labels).value_counts())

new_person = [35,60000,1.0,1.1,0,20000,12.01,0.33,1,1,0]
new_person_array = np.array(new_person).reshape(1, -1)
predicted_cluster = kmeans.predict(new_person_array)
print(predicted_cluster[0])


x_col = credit.columns[5]
y_col = credit.columns[1]
z_col = credit.columns[6]

x_index = credit_numeric.columns.get_loc(x_col)
y_index = credit_numeric.columns.get_loc(y_col)
z_index = credit_numeric.columns.get_loc(z_col)

# Plot 3D clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'red', 'green', 'yellow', 'orange']
for i in range(5):
    ax.scatter(
        credit[labels == i][x_col],
        credit[labels == i][y_col],
        credit[labels == i][z_col],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.25
    )

# Plot centroids
centroids = kmeans.cluster_centers_
ax.scatter(
    centroids[:, x_index],
    centroids[:, y_index],
    centroids[:, z_index],
    c='black', marker='x', s=100, label='Centroids'
)

# Labels and title
ax.set_xlabel(x_col)
ax.set_ylabel(y_col)
ax.set_zlabel(z_col)
ax.set_zlim(0, 30)
ax.set_title('3D KMeans Clustering of Credit Risk Data')
ax.legend()
plt.tight_layout()
plt.show()
