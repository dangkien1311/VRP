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

# start_time = time.time()
# population = None
# def target():
#     nonlocal population
# population = generate_population(vehicle_capacity, customers)
# print(population)
# max_execution_time = 3
# population_thread = threading.Thread(target=target)
# population_thread.start()
# population_thread.join(timeout=max_execution_time)
# elapsed_time = time.time() - start_time
# if elapsed_time > max_execution_time:
#     print(f"Execution time of generate population ({elapsed_time} seconds) exceeded the limit. Rerunning...")
#     python = sys.executable
#     os.execl(python, python, *sys.argv)
# else:
#     print(f"Execution time: {elapsed_time} seconds")
import random


s = 'TABU SEARCH, SIMULATED ANNEALING,GUIDED LOCAL SEARCH'
print(random.random())