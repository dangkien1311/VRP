import math
import random
import numpy as np
import datetime

num_vehicles = 3
vehicle_capacity = 10
num_customers = 100


with open('data.txt', 'r') as file:
    lines = file.readlines()

# Initialize the customers dictionary
customers = {}

# Process each line and create the dictionary entries
for line in lines[1:]:  # Skip the header line
    data = line.strip().split()
    cust_no = int(data[0])
    xcoord = int(data[1])
    ycoord = int(data[2])
    demand = int(data[3])
    ready_time = int(data[4])
    due_date = int(data[5])
    service_time = int(data[6])
    
    customers[cust_no] = (cust_no,xcoord, ycoord, demand, ready_time, due_date, service_time)

# Genetic algorithm parameters
population_size = 100
generations = 200
mutation_rate = 0.1

def calculate_distance(coord1, coord2):
    x_diff = coord1[1] - coord2[1]
    y_diff = coord1[2] - coord2[2]
    distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
    return distance

# Calculate the total cost of a solution
def solution_cost(solution):
    total_cost = 0
    current_time = 0
    current_load = 0
    current_location = 0
    end_point = 0
    for i in solution:
        total_cost += 5 * calculate_distance(customers[i],customers[current_location])
        current_location = i
    total_cost += 5 * calculate_distance(customers[current_location], customers[end_point])
    return total_cost


# Create an initial random population
def generate_population(size):
    population_with_value = []
    population = [random.sample(range(1, num_customers + 1), num_customers - 1) for _ in range(size)]
    for citizen in population:
        value = solution_cost(citizen)
        population_with_value.append({
            'route' : citizen,
            'cost': value
            })
    return population_with_value

# Perform crossover between two parents to create two children
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + [customer for customer in parent2 if customer not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [customer for customer in parent1 if customer not in parent2[:crossover_point]]
    return child1, child2

# Mutate a solution by swapping two customers
def mutate(solution):
    mutated_solution = solution[:]
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(mutated_solution)), 2)
        mutated_solution[idx1], mutated_solution[idx2] = mutated_solution[idx2], mutated_solution[idx1]
    return mutated_solution

# Main genetic algorithm loop
def genetic_algorithm():
    count_solution = 0
    population = generate_population(population_size)
    lowest_cost_element = 0

    for _ in range(generations):
        number_parents = len(population)
        new_population = []
        if number_parents % 2 == 1:
            lowest_cost_element = min(population, key=lambda x: x['cost'])
            population.remove(lowest_cost_element)
            new_population.append(lowest_cost_element)
        for _ in range(number_parents // 2):
            total_costs = [x['cost'] for x in population]
            average_total_cost = sum(total_costs) / len(total_costs)
            # Filter elements with total cost less than average total cost
            filtered_elements = [x for x in population if x['cost'] <= average_total_cost]
            if len(filtered_elements) == 0:
                filtered_elements = [x for x in population if x['cost'] <= average_total_cost + 10]
            if len(filtered_elements) == 1:
                return filtered_elements[0]
            parents = random.sample(filtered_elements, 2)
            infant1, infant2 = crossover(parents[0]['route'], parents[1]['route'])
            infant1 = mutate(infant1)
            infant2 = mutate(infant2)
            cost_of_child1 = solution_cost(infant1)
            cost_of_child2 = solution_cost(infant2)
            child1 = {
                'route' : infant1,
                'cost': cost_of_child1
            }
            child2 = {
                'route' : infant2,
                'cost': cost_of_child2
            }
            new_population.extend([child1, child2])
        
        population = new_population

    unique_list = []
    for item in population:
        if item not in unique_list:
            unique_list.append(item)

    print(unique_list)
    best_solution_i = min(unique_list, key=lambda x: x['cost'])
    return best_solution_i

# Run the genetic algorithm
if __name__ == "__main__":
    start = datetime.datetime.now()
    best_solution = genetic_algorithm()
    print(f"Best solution: ", best_solution['route'])
    print(f"Best cost: ", best_solution['cost'])
    processtime = datetime.datetime.now() - start
    print(f"process time: {processtime}")