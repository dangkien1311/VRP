def find_all_routes(data_list):
    routes = []
    pre_value = -1
    current_route = []
    while True:
        if len(routes) == 4:
            break
        for i in data_list:
            if pre_value == 0:
                # current_route.append(0) 
                routes.append(current_route.copy())
                print(routes)
                current_route.clear()
                pre_value = -1
                break
            if i[0] == 0 and len(current_route) == 0:
                current_route.extend([i[0], i[1]])
                pre_value = i[1]
                data_list.remove(i)
                print(current_route)
                continue
            elif i[0] == pre_value:
                current_route.append(i[1])
                pre_value = i[1]
                data_list.remove(i)
                print(current_route)
                if len(data_list) == 0:
                    routes.append(current_route)
                    break
                continue
    return routes

# Example usage:
data_list = [(0, 3), (0, 5), (0, 6), (0, 8), (1, 0), (2, 0), (3, 1), (4, 0), (5, 4), (6, 10), (7, 9), (8, 7), (9, 2), (10, 0)]
result_routes = find_all_routes(data_list)
for i, route in enumerate(result_routes, 1):
    print(f"Route {i}: {route}")
