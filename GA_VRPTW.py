import csv
from itertools import zip_longest
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

with open('solomon_data.txt', 'r') as file:
    lines = file.readlines()

# Initialize the customers dictionary
customers = {}
num_customers = 100
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
population_size = 500
generations = 2500
mutation_rate = 0.07
pcv = 0.8
time_window_deviation = 50

pr_best_pA = -1
pr_best_pB = -1
fitness_list = {}
chart_punish = {}
chart_wait = {}

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters)
clus  = kmeans.fit(cord_data)
centroids = kmeans.cluster_centers_
clustered_data = [{} for _ in range(num_clusters)]
for i, label in enumerate(kmeans.labels_):
    if i != 0:
        clustered_data[label][i] = customers[i]

def calculate_distance(coord1, coord2):
    x_diff = coord1[1] - coord2[1]
    y_diff = coord1[2] - coord2[2]
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance

def split_route(route):
    total_demand = 0
    demand_groups = []
    current_group = []
    service_time = 0
    cus_before = 0
    for i in route:
        if service_time == 0:
            # service_time = random.randint(customers[i][4],customers[i][5])
            service_time = customers[i][4]
        if service_time != 0:
            service_time = service_time + customers[cus_before][6] + calculate_distance(customers[i],customers[cus_before])
        if total_demand + customers[i][3] <= vehicle_capacity and (service_time >= (customers[i][4] - time_window_deviation) and service_time <= (customers[i][5] + time_window_deviation)):
            current_group.append(i)
            total_demand += customers[i][3]
        else:
            demand_groups.append(current_group)
            current_group = [i]
            total_demand = customers[i][3]
            service_time = 0
        cus_before = i
    demand_groups.append(current_group)
    return demand_groups

# Calculate the total cost of a solution
def solution_cost(solution):
    fn_cost = 0
    wait = 0
    punish = 0
    for route in solution:
        total_cost = 0
        current_location = 0
        end_point = 0
        service_time = 0
        for cust in route:
            if service_time != 0:
                service_time = service_time + customers[cus_before][6] + calculate_distance(customers[cust],customers[cus_before])
            if service_time == 0 :
                # service_time = random.randint(customers[cust][4],customers[cust][5])
                service_time = customers[cust][4]
            if service_time < customers[cust][4]:
                wait += (customers[cust][4] - service_time)
            if service_time > customers[cust][5]:
                punish += (service_time - customers[cust][5])
            total_cost += calculate_distance(customers[current_location],customers[cust])
            current_location = cust
            cus_before = cust
        total_cost += calculate_distance(customers[current_location], customers[end_point])
        fn_cost+=total_cost
    raw_cost = fn_cost
    fn_cost = fn_cost + wait + punish 
    return {'total_cost' : fn_cost, 'wait' : wait, 'punish' : punish, 'cost': raw_cost}


# Create an initial random population
def generate_population(capacity,set_customer,population_size):
    population = []
    for _ in range(population_size):
        wait = 0
        punish = 0
        route_list = []
        list_filtered_route = []
        while True:
            service_time  = 0
            capacity_value = capacity
            filtered_customers = []
            customers_to_remove_set = set(customer_id for sublist in list_filtered_route for customer_id in sublist)
            keys_to_shuffle = [key for key,value in set_customer.items() if key not in customers_to_remove_set]
            random.shuffle(keys_to_shuffle)
            lst_filtered_cust = {key: set_customer[key] for key in keys_to_shuffle if key != 0}
            # if len(lst_filtered_cust) == 1:
            #     list_filtered_route = []
            #     continue
            if len(lst_filtered_cust) == 0:
                route_list.extend(list_filtered_route)
                break
            for customer_id, customer_values in lst_filtered_cust.items():
                if service_time != 0:
                    service_time = service_time + cus_before[6] + calculate_distance(customer_values,cus_before)
                if service_time == 0 :
                    # service_time = random.randint(customer_values[4],customer_values[5])
                    service_time = customer_values[4]
                if customer_values[3] <= capacity_value and (service_time >= (customer_values[4] - time_window_deviation) and service_time <= (customer_values[5] + time_window_deviation)):
                    filtered_customers.append(customer_id)
                    capacity_value -= customer_values[3]
                cus_before = customer_values
            if len(filtered_customers) == 0:
                continue
            list_filtered_route.append(filtered_customers)
        flattened_list = []
        [flattened_list.extend(sublist) for sublist in route_list]
        total_value = solution_cost(split_route(flattened_list))
        flattened_list.append(total_value['total_cost'])
        population.append(flattened_list)
    return population

def calculate_split_cost(parent):
        for i in parent:
            cost_i = solution_cost([i])
            i.append(cost_i['total_cost'])
        return parent

# OX crossover need repair
def crossover(parentA, parentB):
    parentA = parentA[:-1]
    parentB = parentB[:-1]
    childs = []
    index = 0
    for _ in range(0,2):
        if index ==0:
            positions = random.sample(range(0,len(parentA)), 2)
            positions.sort()
            child = [0] * len(parentA)
            child[positions[0]:positions[1]] = parentA[positions[0]:positions[1]]
            p2genes = [gene for gene in parentB if gene not in child]
            child[:positions[0]] = p2genes[:positions[0]]
            child[positions[1]:] = p2genes[positions[0]:]
            child = heuristic_scramble_mutation(child)
            cost = solution_cost(split_route(child))
            child.append(cost['total_cost'])
            childs.append(child)
        else:
            positions = random.sample(range(0,len(parentB)), 2)
            positions.sort()
            child = [0] * len(parentB)
            child[positions[0]:positions[1]] = parentB[positions[0]:positions[1]]
            p2genes = [gene for gene in parentA if gene not in child]
            child[:positions[0]] = p2genes[:positions[0]]
            child[positions[1]:] = p2genes[positions[0]:]
            child = heuristic_scramble_mutation(child)
            cost = solution_cost(split_route(child))
            child.append(cost['total_cost'])
            childs.append(child)
        index +=1
    return childs

# heuristic OX crossover 
def heuristic_crossover(parentA, parentB):
    parentA = parentA[:-1]
    parentB = parentB[:-1]
    childs = []
    split_parentA = calculate_split_cost(split_route(parentA))
    split_parentB = calculate_split_cost(split_route(parentB))
    index = 0
    global pr_best_pA
    global pr_best_pB
    for _ in range(0,2):
        if index == 0:
            child = [0] * len(parentA)
            best_pA = min(split_parentA, key= lambda x:x[-1])
            c_best_pA = best_pA[-1]
            if pr_best_pA >= c_best_pA:
                positions = random.sample(range(0,len(parentA)), 2)
                positions.sort()
                child[positions[0]:positions[1]] = parentA[positions[0]:positions[1]]
                p2genes = [gene for gene in parentB if gene not in child]
                child[:positions[0]] = p2genes[:positions[0]]
                child[positions[1]:] = p2genes[positions[0]:]
                child = mutate(child)
                cost = solution_cost(split_route(child))
                child.append(cost['total_cost'])
                childs.append(child)
            else:
                best_pA = best_pA[:-1]
                positionA1 = parentA.index(best_pA[0])
                positionA2 = parentA.index(best_pA[-1])
                if positionA1 == positionA2:
                    child[positionA1] = best_pA[0]
                else:
                    child[positionA1:positionA2] = best_pA
                p2genesA = [gene for gene in parentB if gene not in child] 
                child[:positionA1] = p2genesA[:positionA1]
                child[positionA2 + 1:] = p2genesA[positionA1:]
                child = mutate(child)
                cost = solution_cost(split_route(child))
                child.append(cost['total_cost'])
                childs.append(child)
            pr_best_pA = c_best_pA
        else:
            child = [0] * len(parentB)
            best_pB = min(split_parentB, key= lambda x:x[-1])
            c_best_pB = best_pB[-1]
            if pr_best_pB >= c_best_pB:
                positions = random.sample(range(0,len(parentB)), 2)
                positions.sort()
                child[positions[0]:positions[1]] = parentB[positions[0]:positions[1]]
                p2genes = [gene for gene in parentA if gene not in child]
                child[:positions[0]] = p2genes[:positions[0]]
                child[positions[1]:] = p2genes[positions[0]:]
                child = mutate(child)
                cost = solution_cost(split_route(child))
                child.append(cost['total_cost'])
                childs.append(child)
            else:
                best_pB = best_pB[:-1]
                positionB1 = parentB.index(best_pB[0])
                positionB2 = parentB.index(best_pB[-1])
                if positionB1 == positionB2:
                    child[positionB1] = best_pB[0]
                else:
                    child[positionB1:positionB2] = best_pB
                p2genesB = [gene for gene in parentA if gene not in child]
                child[:positionB1] = p2genesB[:positionB1]
                child[positionB2 + 1:] = p2genesB[positionB1:]
                child = mutate(child)
                cost = solution_cost(split_route(child))
                child.append(cost['total_cost'])
                childs.append(child)
            pr_best_pB = c_best_pB
        index +=1
    return childs
    
# Mutate a solution by swapping two customers
def mutate(solution):
    mutated_child = solution
    if random.random() < mutation_rate:
        # position = random.randint(0,len(solution))
        # solution = [elem for pair in zip_longest(solution[:position], solution[position:]) for elem in pair if elem is not None]
        ##CIM
        # solution[:position] =  solution[:position][::-1]
        # solution[position:] =  solution[position:][::-1]
        ##swap
        # for i in range(len(solution)):
        #     swap_index1 = random.randint(0, len(solution) - 1)
        #     swap_index2 = random.randint(0, len(solution) - 1)
        #     solution[swap_index1], solution[swap_index2] = solution[swap_index2], solution[swap_index1]
        ## swap segment
        positions = random.sample(range(0,len(solution)), 4)
        positions.sort()
        mutated_child = []
        lst_1 = solution[:positions[0]]
        swap1 = solution[positions[0]:positions[1]]
        lst_2 = solution[positions[1]:positions[2]]
        swap2 = solution[positions[2]:positions[3]]
        lst_3 = solution[positions[3]:]
        mutated_child.extend(lst_1)
        mutated_child.extend(swap2)
        mutated_child.extend(lst_2)
        mutated_child.extend(swap1)
        mutated_child.extend(lst_3)
    return mutated_child

def heuristic_scramble_mutation(solution):
    if random.random() < mutation_rate:
        split_child = calculate_split_cost(split_route(solution))
        worst_genes = max(split_child, key= lambda x:x[-1])
        positionA1 = solution.index(worst_genes[0])
        positionA2 = solution.index(worst_genes[-2]) + 1
        temp_list = solution[positionA1:positionA2]
        random.shuffle(temp_list)
        solution[positionA1:positionA2] = temp_list
    return solution


def generate_newgeneration(parent_generation,preservation_rate):
    new_generation = []
    origin_goal_len = len(parent_generation)
    pop_num = int((origin_goal_len) * preservation_rate)
    for _ in range(0,pop_num):
        elite_child = min(parent_generation, key  = lambda x: x[-1])
        elite_index = parent_generation.index(elite_child)
        del parent_generation[elite_index]
        new_generation.append(elite_child)
    list_parent = sorted(parent_generation, key = lambda x: x[-1])
    cut = int((len(parent_generation))/2)
    list_parent = list_parent[:cut]
    while True:
        #get random two parent
        index = random.sample(range(0,len(list_parent)),2)
        route1 = list_parent[index[0]]
        route2 = list_parent[index[1]]
        new_routes = [route1,route2]
        #crossover two parent
        if random.random() <= pcv:
            new_routes = heuristic_crossover(route1,route2)
        new_generation.append(new_routes[0])
        if len(new_generation) == origin_goal_len:
            break
        new_generation.append(new_routes[1])
        if len(new_generation) == origin_goal_len:
            break
    return new_generation 

        
# Main genetic algorithm loop
def genetic_algorithm():
    fn_solution = []
    for lab,pop in enumerate(clustered_data):
        fitness_list[lab] = []
        chart_punish[lab] = [] 
        chart_wait[lab] = []
        population = generate_population(vehicle_capacity,pop,population_size)
        bestP = min(population, key= lambda x:x[-1])
        fitness_list[lab].append(bestP[-1])
        total_value = solution_cost(split_route(bestP[:-1]))
        chart_punish[lab].append(total_value['punish'])
        chart_wait[lab].append(total_value['wait'])
        i = 1
        print(f'cluster: {lab}')
        for _ in range(generations):
            p = (i/generations) * 100
            if p % 10 == 0:
                print(f"{int(p)}%")
            new_gen = generate_newgeneration(population,0.1)
            bestG = min(new_gen, key= lambda x:x[-1])
            fitness_list[lab].append(bestG[-1])
            total_value = solution_cost(split_route(bestG[:-1]))
            chart_punish[lab].append(total_value['punish'])
            chart_wait[lab].append(total_value['wait'])
            population = new_gen
            i += 1
        best_solution = min(population, key= lambda x:x[-1])
        fn_solution.append(best_solution)
    return fn_solution

def caculate_capacity(route):
    sum_demand = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    return sum_demand

def run_program():
    fn_cost = 0
    fn_number_route = 0
    total_demand = 0
    best_solution = genetic_algorithm()
    for lab,ebest in enumerate(best_solution):
        total_cost = ebest[-1]
        fn_cost += total_cost
        ebest = ebest[:-1]
        print(f'optimization for cluster {lab} : {ebest}')
        result = split_route(ebest)
        fn_route_cost = solution_cost(result)
        fn_number_route += len(result)
        for i in range(0, len(result)):
            print(f"Route {i + 1}:", result[i])
            cap = caculate_capacity(result[i])
            print(f"capacity of route {i + 1}: {cap}")
            total_demand += cap
            route_cost = solution_cost([result[i]])
            print(f"Cost for route {i + 1}:", route_cost['total_cost'])
            print(f"wait for route {i + 1}:", route_cost['wait'])
            print(f"punish for route {i + 1}:", route_cost['punish'])
            print("----------------------------------------------------------------")
        print(f"Total cost: {total_cost}")
        print(f"Total wait: {fn_route_cost['wait']}")
        print(f"Total punish: {fn_route_cost['punish']}")
        ratio = round((1 - ((total_cost)/fitness_list[lab][0])) * 100,2)
        print(f"population in generations {generations} is {ratio}% better than the original one")
        print(f"total_demand {total_demand} ")
        plt.plot(fitness_list[lab])
        plt.plot(fitness_list[lab], label='total_cost')
        plt.plot(chart_punish[lab], label='punish')
        plt.plot(chart_wait[lab], label='wait')
        # Add labels and a title
        plt.xlabel("Generation")
        plt.ylabel("Total cost")
        plt.title("Line chart of total cost per generation")
        plt.legend()
        plt.savefig(f'output{lab}.png')
        plt.close()
    print(f'Cost of all route: {fn_cost}')
    print(f'total route: {fn_number_route}')
    return 0

if __name__ == "__main__":
    start = datetime.datetime.now()
    run_program()
    processtime = datetime.datetime.now() - start
    print(f"process time: {processtime}")


