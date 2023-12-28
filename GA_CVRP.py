import csv
import math
import multiprocessing
import os
import random
import threading
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import sys

with open('data.txt', 'r') as file:
    lines = file.readlines()

# Initialize the customers dictionary
customers = {}
num_customers = 200
end_point_data = num_customers + 2
cord_data = []
# Process each line and create the dictionary entries
for line in lines[1:end_point_data]:  # Skip the header line
    list_cord = []
    data = line.strip().split(',')
    cust_no = int(data[0])
    xcoord = int(data[1])
    ycoord = int(data[2])
    demand = int(data[3])
    ready_time = int(data[4])
    due_date = int(data[5])
    service_time = int(data[6])
    list_cord.append(xcoord)
    list_cord.append(ycoord)
    cord_data.append(list_cord)
    customers[cust_no] = (cust_no,xcoord, ycoord, demand, ready_time, due_date, service_time)

# num_vehicles = 10
vehicle_capacity = 200

population_size = 100
generations = 100
mutation_rate = 0.01
pcv = 0.8


fitness_list = []

def check_condition(route):
    check = True
    sum_demand = sum(customers[cus][3] for cus in route)
    if sum_demand > vehicle_capacity:
        check = False
    return check

def calculate_distance(coord1, coord2):
    x_diff = coord1[1] - coord2[1]
    y_diff = coord1[2] - coord2[2]
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance

def split_route(route):
    total_demand = 0
    demand_groups = []
    current_group = []

    for i in route:
        if total_demand + customers[i][3] <= vehicle_capacity:
            current_group.append(i)
            total_demand += customers[i][3]
        else:
            demand_groups.append(current_group)
            current_group = [i]
            total_demand = customers[i][3]

    if current_group:
        demand_groups.append(current_group)
    return demand_groups

# Calculate the total cost of a solution
def solution_cost(solution):
    fn_cost = 0
    for route in solution:
        total_cost = 0
        current_location = 0
        end_point = 0
        for cust in route:
            total_cost += calculate_distance(customers[current_location],customers[cust])
            current_location = cust
        total_cost += calculate_distance(customers[current_location], customers[end_point])
        fn_cost+=total_cost
    return fn_cost


# Create an initial random population
def generate_population(capacity,set_customer,population_size):
    population = []
    for _ in range(population_size):
        route_list = []
        list_filtered_route = []
        while True:
            capacity_value = capacity
            filtered_customers = []
            customers_to_remove_set = set(customer_id for sublist in list_filtered_route for customer_id in sublist)
            keys_to_shuffle = [key for key,value in set_customer.items() if key not in customers_to_remove_set]
            random.shuffle(keys_to_shuffle)
            lst_filtered_cust = {key: set_customer[key] for key in keys_to_shuffle if key != 0}
            if len(lst_filtered_cust) == 1:
                list_filtered_route = []
                continue
            elif len(lst_filtered_cust) == 0:
                route_list.extend(list_filtered_route)
                break
            for customer_id, customer_values in lst_filtered_cust.items():
                if customer_values[3] <= capacity_value:
                    filtered_customers.append(customer_id)
                    capacity_value -= customer_values[3]
            list_filtered_route.append(filtered_customers)
        flattened_list = []
        [flattened_list.extend(sublist) for sublist in route_list]
        total_value = solution_cost(split_route(flattened_list))
        flattened_list.append(total_value)
        population.append(flattened_list)
    return population

# OX1 crossover 2 point ++
def crossover(parentA, parentB):
    parentA = parentA[:-1]
    parentB = parentB[:-1]
    childs = []
    for _ in range(0,2):
        positions = random.sample(range(0,len(parentA)), 2)
        positions.sort()
        child = [0] * len(parentA)
        child[positions[0]:positions[1]] = parentA[positions[0]:positions[1]]
        p2genes = [gene for gene in parentB if gene not in child]
        child[:positions[0]] = p2genes[:positions[0]]
        child[positions[1]:] = p2genes[positions[0]:]
        child = mutate(child)
        cost = solution_cost(split_route(child))
        child.append(cost)
        childs.append(child)
    return childs

# # OX1 crossover 1 point +
# def crossover(parentA, parentB):
#     parentA = parentA[:-1]
#     parentB = parentB[:-1]
#     childs = []
#     for _ in range(0,2):
#         positions = random.sample(range(0,len(parentA)), 1)
#         child = [0] * len(parentA) 
#         child[:positions[0]] = parentA[:positions[0]]
#         p2genes = [gene for gene in parentB if gene not in child]
#         child[positions[0]:] = p2genes
#         child = mutate(child)
#         cost = solution_cost(split_route(child))
#         child.append(cost)
#         childs.append(child)
#     return childs



# Mutate a solution by swapping two customers
def mutate(solution):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

def generate_newgeneration(parent_generation,remain_ratio):
    new_generation = []
    origin_goal_len = len(parent_generation)
    pop_num = int((origin_goal_len) * remain_ratio)
    for _ in range(0,pop_num):
        elite_child = min(parent_generation, key  = lambda x: x[-1])
        elite_index = parent_generation.index(elite_child)
        del parent_generation[elite_index]
        new_generation.append(elite_child)
    goal_len = len(parent_generation)
    if goal_len % 2 == 1:
            elite_child = min(parent_generation, key  = lambda x: x[-1])
            elite_index = parent_generation.index(elite_child)
            del parent_generation[elite_index]
            new_generation.append(elite_child)
    count_pvc = 0
    list_parent = sorted(parent_generation, key = lambda x: x[-1])
    cut = int((len(parent_generation))/2)
    list_parent = list_parent[:cut]
    while True:
        if len(new_generation) == origin_goal_len:
            break
        #get random two parent
        index = random.sample(range(0,len(list_parent)),2)
        route1 = list_parent[index[0]]
        route2 = list_parent[index[1]]
        new_routes = [route1,route2]
        #crossover two parent
        if random.random() <= pcv:
            new_routes = crossover(route1,route2)
        new_generation.extend(new_routes)
        count_pvc += 1 
    return new_generation 

        
# Main genetic algorithm loop
def genetic_algorithm():
    population = generate_population(vehicle_capacity,customers,population_size)
    bestP = min(population, key= lambda x:x[-1])
    fitness_list.append(bestP[-1])
    gene = 0
    for _ in range(generations):
        # print(gene)
        # print(population)
        # print(f'length of parent: {len(population)}')
        # print("----------------------------------------------------------------")
        new_gen = generate_newgeneration(population,0.2)
        # print(new_gen)
        # print(f'length of child: {len(new_gen)}')
        bestG = min(new_gen, key= lambda x:x[-1])
        fitness_list.append(bestG[-1])
        population = new_gen
        gene += 1
    best_solution = min(population, key= lambda x:x[-1])
    return best_solution

def caculate_capacity(route):
    sum_demand = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    return sum_demand

def run_program():
    total_demand = 0
    best_solution = genetic_algorithm()
    total_cost = best_solution[-1]
    best_solution = best_solution[:-1]
    result = split_route(best_solution)
    print(best_solution)
    print("----------------------------------------------------------------")
    for i in range(0, len(result)):
        print(f"Route {i + 1}:", result[i])
        cap = caculate_capacity(result[i])
        print(f"capacity of route {i + 1}: {cap}")
        total_demand += cap
        print(f"Cost for route {i + 1}:", solution_cost([result[i]]))
        print("----------------------------------------------------------------")
    print(f"Total cost: {total_cost}")
    ratio = round((1 - ((total_cost)/fitness_list[0])) * 100,2)
    print(f"population in generations {generations} is {ratio}% better than the origin one")
    print(f"total_demand {total_demand} ")
    plt.plot(fitness_list)
    # Add labels and a title
    plt.xlabel("Generation")
    plt.ylabel("Total cost")
    plt.title("Line chart of total cost on each generation")

    plt.savefig('output.png')
    return 0

if __name__ == "__main__":
    start = datetime.datetime.now()
    # max_execution_time = 60  # Max execution time in seconds
    # while True:
    #     process = multiprocessing.Process(target=run_program)
    #     start_time = datetime.datetime.now()
    #     process.start()
        
    #     process.join(max_execution_time)  # Wait for the process to complete or timeout
        
    #     elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    #     if elapsed_time >= max_execution_time:
    #         print("Execution time limit reached. Terminating the process.")
    #         process.terminate()
    #         process.join()  # Wait for the process to terminate
    #     else:
    #         print("Execution completed within the time limit.")
            # processtime = datetime.datetime.now() - start
            # print(f"process time: {processtime}")
            # sys.exit()
    run_program()
    processtime = datetime.datetime.now() - start
    print(f"process time: {processtime}")