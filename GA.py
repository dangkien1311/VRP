import csv
import math
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with open('data.txt', 'r') as file:
    lines = file.readlines()

# Initialize the customers dictionary
customers = {}
num_customers = 100
cord_data = []
# Process each line and create the dictionary entries
for line in lines[1:]:  # Skip the header line
    list_cord = []
    data = line.strip().split()
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

num_vehicles = 3
vehicle_capacity = 80

population_size = 4
generations = 100
mutation_rate = 0.1


fitness_list = []

def check_condition(route):
    check = True
    sum_demand = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    if sum_demand > vehicle_capacity:
        check = False
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
        total_cost += 5 * calculate_distance(customers[current_location],customers[cust])
        current_location = cust
    total_cost += 5 * calculate_distance(customers[current_location], customers[end_point])

    return total_cost


# Create an initial random population
def generate_population(capacity,set_customer):
    set_route = []
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
            if customer_id == 0:
                continue
            if customer_values[3] <= capacity_value:
                filtered_customers.append(customer_id)
                capacity_value -= customer_values[3]
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
    # Generate random positions for slides
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
    return new_route1, new_route2


# Mutate a solution by swapping two customers
def mutate(solution):
    mutated_solution = solution[:]
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated_solution)), 2)
        mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]
    return mutated_solution


def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
    
def is_route_exists(route, generation):
    for item in generation:
        if item['route'] == route:
            return True
    return False
   
def generate_newgeneration(parent_generation):
    new_generation = []
    goal_len = len(parent_generation)
    
    if goal_len % 2 == 1:
            elite_child = min(parent_generation, key  = lambda x: x['cost'])
            elite_index = parent_generation.index(elite_child)
            del parent_generation[elite_index]
            new_generation.append(elite_child)

    while True:
        list_parent = [i for i in range(len(parent_generation))]

        if len(new_generation) == goal_len:
            break
        #get random two parent

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
        if sum_cost < (value1 + value2):
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
    
    return new_generation 

# Main genetic algorithm loop
def genetic_algorithm():
    population = generate_population(vehicle_capacity,customers)
    print(population)
    fitness_list.append(population[-1]['total_cost'])
    gene = 0
    list_solution = []
    for _ in range(generations):
        print(gene)
        # try:
        print(population)
        print("--------------------------------")
        temp_populaiton = population[:-1]
        new_gen = generate_newgeneration(temp_populaiton)
        total_cost = sum(entry['cost'] for entry in new_gen)
        new_gen.append({'total_cost' : total_cost})
       
        print(new_gen)
        fitness_list.append(total_cost)
        list_solution.append(new_gen)
        population = new_gen
        # except Exception as e:
        #     print(e)
            # break
        gene += 1
    best_solution =  min(list_solution, key=lambda x: x[-1]['total_cost'])
    print(best_solution)
    return best_solution

# Run the genetic algorithm
def caculate_capacity(route):
    sum_demand = 0
    sum_demand = sum(customers[cus][3] for cus in route)
    return sum_demand
if __name__ == "__main__":
    start = datetime.datetime.now()
    best_solution = genetic_algorithm()
    print(best_solution)
    for i in range(0, len(best_solution) -1):
        print(f"Route {i + 1}:", best_solution[i]['route'])
        cap = caculate_capacity(best_solution[i]['route'])
        print(f"capacity of route {i + 1}: {cap}")
        print(f"Cost for route {i + 1}:", best_solution[i]['cost'])
    processtime = datetime.datetime.now() - start
    print(f"process time: {processtime}")
    plt.plot(fitness_list)
    # Add labels and a title
    plt.xlabel("X-axis (Index)")
    plt.ylabel("Y-axis (Value)")
    plt.title("Line chart of best solution in each generation")

    # Show the chart
    plt.show()
