# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Tạo dữ liệu mẫu
# with open('data.txt', 'r') as file:
#     lines = file.readlines()

# total_demand = 0
# cord_data = []
# # Process each line and create the dictionary entries
# for line in lines[2:]:  # Skip the header line
#     list_cord = []
#     data = line.strip().split()
#     cust_no = int(data[0])
#     xcoord = int(data[1])
#     ycoord = int(data[2])
#     demand = int(data[3])
#     total_demand += demand
#     ready_time = int(data[4])
#     due_date = int(data[5])
#     service_time = int(data[6])
#     list_cord.append(xcoord)
#     list_cord.append(ycoord)
#     cord_data.append(list_cord)
    
# num_clusters = 3
# # Xây dựng mô hình K-Means với n cụm
# kmeans = KMeans(n_clusters=num_clusters)
# # Thực hiện gom cụm dữ liệu
# clus  = kmeans.fit(cord_data)
# # Lấy vị trí của các trung tâm cụm
# centroids = kmeans.cluster_centers_

# clustered_data = [[] for _ in range(num_clusters)]

# for i, label in enumerate(kmeans.labels_):
#     clustered_data[label].append(cord_data[i])

# print(clustered_data)
# # Hiển thị kết quả
# for i, cluster in enumerate(clustered_data):
#     print(f"Cụm {i+1}:")
#     print(centroids[i])
#     for point in cluster:
#         print(point)
#     print()

# plt.figure(figsize=(12, 8))

# for cluster, color in zip([clustered_data[0], clustered_data[1],clustered_data[2],[[40,50]]], ['r', 'g', 'b','m']):
#     data = list(zip(*cluster))
#     plt.scatter(data[0], data[1], c=color, label=f'Cluster {color}')

# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Biểu đồ dữ liệu của các cụm')
# plt.legend()
# plt.show()

[9, 4, 6, 2, 1, 5, 3, 7, 8, 10, 211.15436332297512]
A = [22, 43, 62, 25, 131, 103, 140, 176, 9, 104, 12, 95, 71, 127, 157, 119, 39, 196, 31, 129, 175, 45, 137, 10, 112, 150, 116, 105, 152, 76, 38, 134, 1, 100, 4, 17, 46, 41, 115, 179, 168, 183, 73, 85, 68, 91, 126, 156, 172, 122, 164, 197, 88, 191, 21, 193, 139, 169, 52, 194, 162, 34, 53, 170, 48, 147, 120, 163, 144, 130, 161, 23, 200, 121, 187, 198, 138, 186, 64, 195, 14, 149, 51, 94, 30, 29, 49, 185, 189, 66, 72, 47, 111, 69, 106, 74, 109, 75, 6, 19, 89, 178, 78, 108, 80, 101, 180, 154, 96, 148, 123, 26, 151, 50, 15, 27, 36, 98, 28, 173, 158, 114, 58, 166, 153, 145, 60, 59, 61, 97, 188, 81, 110, 192, 84, 93, 56, 146, 99, 32, 125, 63, 102, 155, 83, 184, 124, 136, 107, 190, 181, 141, 42, 86, 132, 24, 35, 5, 128, 77, 159, 182, 67, 165, 3, 70, 11, 143, 133, 44, 18, 55, 199, 57, 117, 118, 174, 167, 54, 40, 160, 33, 65, 87, 92, 79, 177, 2, 142, 7, 135, 37, 8, 13, 90, 82, 171, 16, 113, 20]
A.sort()
B = [i for i in range(1, 201)]
if A==B:
    print('ok')
