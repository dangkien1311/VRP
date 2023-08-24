sample_list = [
    {'sample': [2, 4, 5, 3, 1], 'total': 15},
    {'sample': [2, 5, 3, 4, 1], 'total': 15},
    {'sample': [2, 5, 1, 4, 3], 'total': 15},
    {'sample': [4, 2, 5, 3, 1], 'total': 10}
]

# Find the element with the lowest total cost
lowest_cost_element = min(sample_list, key=lambda x: x['total'])

print("Element with lowest cost:", lowest_cost_element)
print(type(sample_list[0]['sample']))