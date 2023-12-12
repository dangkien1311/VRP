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
num_customers = 10
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

population_size = 4
generations = 100
mutation_rate = 0.01
pcv = 0.8


fitness_list = []

def check_condition(route):
    check = True
    start_time_service = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    start_time_service = customers[route[0]][4]
    if sum_demand > vehicle_capacity:
        check = False
    for i in range(0,len(route) - 1):
        if not (start_time_service >= customers[route[i]][4] - 50 and start_time_service <= customers[route[i]][5] + 50):
            check = False
        start_time_service = start_time_service + (calculate_distance(customers[route[i]],customers[route[i+1]])) + customers[route[i]][6]
    return check

def calculate_distance(coord1, coord2):
    x_diff = coord1[1] - coord2[1]
    y_diff = coord1[2] - coord2[2]
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance


# Calculate the total cost of a solution
def solution_cost(solution):
    total_cost = 0
    current_location = 0
    end_point = 0
    for cust in solution:
        total_cost += calculate_distance(customers[current_location],customers[cust])
        current_location = cust
    total_cost += calculate_distance(customers[current_location], customers[end_point])
    return total_cost


# Create an initial random population
def generate_population(capacity,set_customer):
    set_route = []
    route_list = []
    list_filtered_route = []
    punish = 0
    start_time = time.time()
    while True:
        start_time_service = 0
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
        elapsed_time = time.time() - start_time
        if elapsed_time > 3:
            start_time = time.time()
            print("regenerate...")
            list_filtered_route = []
            continue
        for customer_id, customer_values in lst_filtered_cust.items():
            keys_list = list(lst_filtered_cust.keys())
            # print(keys_list)
            index = keys_list.index(customer_id) + 1
            if len(filtered_customers) == 0:
                start_time_service = customer_values[4]
            if customer_values[3] <= capacity_value and (start_time_service >= customer_values[4] and start_time_service <= customer_values[5]):
                filtered_customers.append(customer_id)
                # if start_time_service < customer_values[4] and start_time_service >= customer_values[4] - 50:
                #     punish += (customer_values[4] - start_time_service)
                # if start_time_service > customer_values[5] and start_time_service <= customer_values[5] + 50:
                #     punish += abs(customer_values[5] - start_time_service)
                try:
                    start_time_service = start_time_service + (calculate_distance(customers[customer_id],customers[keys_list[index]])) + customer_values[6]
                except:
                    break
                if(start_time_service > customers[0][5]):
                    break
                capacity_value -= customer_values[3]
        if len(filtered_customers) == 1:
            continue
        list_filtered_route.append(filtered_customers)
    population_with_value = []
    total_cost = 0
    for citizen in route_list:
        value = solution_cost(citizen)
        population_with_value.append({
            'route' : citizen,
            'cost': value
            })
        total_cost += value
    population_with_value.append({'total_cost' : total_cost})
    set_route.extend(population_with_value)
    return set_route

def crossover(route1, route2):
    min_len = min(len(route1),len(route2))
    positions = random.sample(range(0, min_len), 2)
    positions.sort()
    new_route1 = route1.copy()
    new_route2 = route2.copy()
    start_index = 0
    for i in positions:
        for _ in range(start_index,i):
            index1 = random.randint(start_index, i)
            index2 = random.randint(start_index, i)
            new_route1[index1], new_route2[index2] = new_route2[index2], new_route1[index1]
        start_index = i
    new_route1 = mutate(new_route1)
    new_route2 = mutate(new_route2)
    return new_route1, new_route2
    


# Mutate a solution by swapping two customers
def mutate(solution):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(solution)), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

   
def generate_newgeneration(parent_generation,remain_ratio):
    new_generation = []
    origin_goal_len = len(parent_generation)
    pop_num = int(origin_goal_len * remain_ratio)
    for _ in range(0,pop_num):
        elite_child = min(parent_generation, key  = lambda x: x['cost'])
        elite_index = parent_generation.index(elite_child)
        del parent_generation[elite_index]
        new_generation.append(elite_child)
    goal_len = len(parent_generation)
    if goal_len % 2 == 1:
            elite_child = min(parent_generation, key  = lambda x: x['cost'])
            elite_index = parent_generation.index(elite_child)
            del parent_generation[elite_index]
            new_generation.append(elite_child)
    count_pvc = 0
    acp_cros = int((len(parent_generation)/2) * pcv)
    while True:
        list_parent = [i for i in range(len(parent_generation))]
        if len(new_generation) == origin_goal_len:
            break
        #get random two parent
        if count_pvc == acp_cros:
            for i in parent_generation:
                new_generation.append(i)
            break
        index = random.sample(list_parent,2)
        
      
        route1 = parent_generation[index[0]]['route']
        route2 = parent_generation[index[1]]['route']
        #crossover two parent
        new_route1, new_route2 = crossover(route1,route2)

        #check condition of two new child
        if check_condition(new_route1) == False or check_condition(new_route2) == False:
            continue

        #caculate sum to check if sum cost if next gen is better than previous one
        sum_cost = parent_generation[index[0]]['cost'] + parent_generation[index[1]]['cost']
        value1 = solution_cost(new_route1)
        value2 = solution_cost(new_route2)
        # print(new_route1,new_route2)
        # print(sum_cost)
        # print(value1 + value2)
        # time.sleep(1)
        if sum_cost <= (value1 + value2):
            continue

        #delete parent
        if index[0] > index[1]:
            del parent_generation[index[0]]
            del parent_generation[index[1]]
        else:
            del parent_generation[index[1]]
            del parent_generation[index[0]]

        new_generation.extend([
            {
                'route' : new_route1,
                'cost' : value1
            },
            {
                'route' : new_route2,
                'cost' : value2
            }
        ])
        count_pvc += 1 
        # print(f" new gen added {len(new_generation)} / {goal_len}")
        # time.sleep(1)
    return new_generation 

        
# Main genetic algorithm loop
def genetic_algorithm():
    population = generate_population(vehicle_capacity, customers)
    print(population)
    fitness_list.append(population[-1]['total_cost'])
    gene = 0
    list_solution = []
    for _ in range(generations):
        print(gene)
        try:
            print(population)
            print("----------------------------------------------------------------")
            temp_populaiton = population[:-1]
            new_gen = generate_newgeneration(temp_populaiton,0.2)
            total_cost = sum(entry['cost'] for entry in new_gen)
            new_gen.append({'total_cost' : total_cost})
            print(new_gen)
            fitness_list.append(total_cost)
            list_solution.append(new_gen)
            population = new_gen
        except Exception as e:
            print(e)
            break
        gene += 1
    best_solution = list_solution[-1]
    print(best_solution)
    return best_solution

def caculate_capacity(route):
    sum_demand = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    return sum_demand

def run_program():
    total_demand = 0
    best_solution = genetic_algorithm()
    print(best_solution)
    print("----------------------------------------------------------------")
    for i in range(0, len(best_solution) -1):
        print(f"Route {i + 1}:", best_solution[i]['route'])
        cap = caculate_capacity(best_solution[i]['route'])
        print(f"capacity of route {i + 1}: {cap}")
        total_demand += cap
        print(f"Cost for route {i + 1}:", best_solution[i]['cost'])
        print("----------------------------------------------------------------")
    print(f"Total cost: {best_solution[-1]['total_cost']}")
    ratio = round((1 - ((best_solution[-1]['total_cost'])/fitness_list[0])) * 100,2)
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
    max_execution_time = 60  # Max execution time in seconds
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