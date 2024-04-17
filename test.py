import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
from itertools import zip_longest

# Tạo dữ liệu mẫu
with open('solomon_data.txt', 'r') as file:
    lines = file.readlines()

total_demand = 0
cord_data = []
# Process each line and create the dictionary entries
for line in lines[1:202]:  # Skip the header line
    list_cord = []
    data = line.strip().split(',')
    cust_no = int(data[0])
    xcoord = int(data[1])
    ycoord = int(data[2])
    demand = int(data[3])
    total_demand += demand
    ready_time = int(data[4])
    due_date = int(data[5])
    service_time = int(data[6])
    list_cord.append(ready_time)
    list_cord.append(due_date)
    cord_data.append(list_cord)
    
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters)
clus  = kmeans.fit(cord_data)
centroids = kmeans.cluster_centers_

clustered_data = [[] for _ in range(num_clusters)]

print(kmeans.labels_)

for i, label in enumerate(kmeans.labels_):
    clustered_data[label].append(cord_data[i])

# for i, cluster in enumerate(clustered_data):
#     print(f"Cụm {i+1}:")
#     print(centroids[i])
#     for point in cluster:
#         print(point)
#     print()

plt.figure(figsize=(12, 8))

for cluster, color in zip([clustered_data[0], clustered_data[1],clustered_data[2],clustered_data[3],clustered_data[4],clustered_data[5],clustered_data[6],clustered_data[7],clustered_data[8],clustered_data[9],[[70,70]]], ['red', 'brown', 'blue','green','orange','purple','olive','tan','maroon','yellow','black']):
    data = list(zip(*cluster))
    plt.scatter(data[0], data[1], c=color, label=f'Cluster {color}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Biểu đồ dữ liệu của các cụm')
plt.legend()
plt.show()
