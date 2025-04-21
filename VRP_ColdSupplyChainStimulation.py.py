# Vehicle Routing Problem (VRP) for Cold Chain Logistics with Simulated Euclidean Distances

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# INPUT PARAMETER
v = 35  # speed in km/h
c_cd = 645000
c_vc = 3000
c_t = 30000
c_s = 45000
p = 15000
phi_1 = 0.05
phi_2 = 0.075
q_k = 2000

# INPUT DATA
n_customers = 5
nodes = ['Depot'] + [f'Store_{i+1}' for i in range(n_customers)]
coords = {
    'Node': nodes,
    'x': [0, 6, 15, 12, 3, 18],
    'y': [0, 7, 14, 5, 13, 9],
    'demand': [0, 620, 640, 780, 810, 600],
    'service_time': [0, 0.13, 0.14, 0.17, 0.18, 0.13]
}
df = pd.DataFrame(coords)
n = len(df)

# Compute distance and time matrices
distance = np.zeros((n, n))
time = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist = np.linalg.norm(df.loc[i, ['x', 'y']] - df.loc[j, ['x', 'y']])
        distance[i, j] = dist
        time[i, j] = dist / v

# BUILDING MODEL
mdl = Model("VRP_ColdChain_Euclidean")
m = 3  # number of vehicles

x = mdl.binary_var_cube(n, n, m, name="x")
f = mdl.continuous_var_cube(n, n, m, name="f")
delta = mdl.continuous_var_cube(n, n, m, lb=0, name="delta")

# Objective function (linear approximation)
cp1 = mdl.sum(c_cd * x[0, j, k] for j in range(1, n) for k in range(m))
cp2 = mdl.sum(c_vc * distance[i, j] * x[i, j, k] for i in range(n) for j in range(n) if i != j for k in range(m))
cp3 = mdl.sum(x[i, j, k] * (c_t * time[i, j] + c_s * df.loc[j, 'service_time']) for i in range(n) for j in range(n) if i != j for k in range(m))
cp4 = mdl.sum(
    x[i, j, k] * p * (
        df.loc[j, 'demand'] * phi_1 * time[i, j] +
        delta[i, j, k] * phi_2 * df.loc[j, 'service_time']
    )
    for i in range(n) for j in range(n) if i != j for k in range(m)
)
mdl.minimize(cp1 + cp2 + cp3 + cp4)

# Constraints
# (C1) Each customer must be visited exactly once by one vehicle
for j in range(1, n):
    mdl.add_constraint(mdl.sum(x[i, j, k] for i in range(n) if i != j for k in range(m)) == 1)
# (C2) Capacity constraint: total demand served by vehicle k must not exceed q_k
for k in range(m):
    mdl.add_constraint(mdl.sum(df.loc[j, 'demand'] * x[i, j, k] for i in range(n) for j in range(1, n) if i != j) <= q_k)
    # (3) Each vehicle must depart from and return to the depot exactly once
    mdl.add_constraint(mdl.sum(x[0, j, k] for j in range(1, n)) == 1)
    mdl.add_constraint(mdl.sum(x[j, 0, k] for j in range(1, n)) == 1)
    # (4) Flow conservation: number of entries = number of exits for each node (except depot)
    for h in range(1, n):
        mdl.add_constraint(mdl.sum(x[i, h, k] for i in range(n) if i != h) == mdl.sum(x[h, j, k] for j in range(n) if j != h))
# (5) Flow constraints: ensure valid transported quantities if route is used
for i in range(n):
    for j in range(n):
        if i != j:
            for k in range(m):
                mdl.add_constraint(f[i, j, k] >= df.loc[j, 'demand'] * x[i, j, k])
                mdl.add_constraint(f[i, j, k] <= q_k * x[i, j, k])
                mdl.add_constraint(delta[i, j, k] >= f[i, j, k] - df.loc[j, 'demand'])
                mdl.add_constraint(delta[i, j, k] <= q_k * x[i, j, k])

# (6) Demand satisfaction constraint:
#  inflow - outflow of goods at customer j must equal its demand (only if visited)
for j in range(1, n):
    for k in range(m):
        mdl.add_constraint(mdl.sum(f[i, j, k] for i in range(n) if i != j) -
                           mdl.sum(f[j, i, k] for i in range(n) if i != j) ==
                           df.loc[j, 'demand'] * mdl.sum(x[i, j, k] for i in range(n) if i != j))

# SOLVE
solution = mdl.solve(log_output=True)

# PRINTING OUTPUT
if solution:
    print("\nTotal Cost:", round(mdl.objective_value, 2))

    route_table = PrettyTable()
    route_table.field_names = ["Vehicle", "Route", "Total Load (kg)", "Total Travel Time (hr)", "Total Service Time (hr)"]

    transfer_table = PrettyTable()
    transfer_table.field_names = ["Vehicle", "From", "To", "Load (kg)", "Travel Time (hr)", "Service Time (hr)"]

    for k in range(m):
        route = []
        total_load = 0
        total_travel_time = 0.0
        total_service_time = 0.0
        current = 0
        visited = set()

        while True:
            next_node = None
            for j in range(n):
                if j != current and x[current, j, k].solution_value > 0.5 and j not in visited:
                    next_node = j
                    break
            if next_node is None:
                break

            load_val = round(f[current, next_node, k].solution_value, 1)
            travel_time = round(time[current, next_node], 4)
            service_time = round(df.loc[next_node, 'service_time'], 4)

            transfer_table.add_row([k+1, nodes[current], nodes[next_node], load_val, travel_time, service_time])

            total_load += df.loc[next_node, 'demand']
            total_travel_time += travel_time
            total_service_time += service_time

            route.append(next_node)
            visited.add(next_node)
            current = next_node

        # Return to depot
        if visited and current != 0:
            time_back = round(time[current, 0], 4)
            transfer_table.add_row([k+1, nodes[current], "Depot", 0.0, time_back, 0.0])
            total_travel_time += time_back
            route.append(0)

        if route:
            route_str = "0-" + "-".join(str(r) for r in route)
            route_table.add_row([k+1, route_str, total_load, round(total_travel_time, 4), round(total_service_time, 4)])

    print("\nBảng 1. Thứ tự giao hàng của các xe tải lạnh")
    print(route_table)

    print("\nBảng 2. Lượng vận chuyển, thời gian di chuyển và phục vụ từng chặng")
    print(transfer_table)

# PLOT 2D ROUTES
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], color='black', s=100, label='Stores/Depot')
for i, row in df.iterrows():
    plt.text(row['x'] + 0.3, row['y'], row['Node'], fontsize=10)

for k in range(m):
    current = 0
    visited = set()
    route_color = colors[k % len(colors)]
    while True:
        next_node = None
        for j in range(n):
            if j != current and x[current, j, k].solution_value > 0.5 and j not in visited:
                next_node = j
                break
        if next_node is None:
            break
        x1, y1 = df.loc[current, ['x', 'y']]
        x2, y2 = df.loc[next_node, ['x', 'y']]
        dx, dy = x2 - x1, y2 - y1
        plt.arrow(x1, y1, dx, dy, head_width=0.4, length_includes_head=True,
                  color=route_color, label=f"Vehicle {k+1}" if len(visited) == 0 else "")
        visited.add(next_node)
        current = next_node

    if current != 0:
        x1, y1 = df.loc[current, ['x', 'y']]
        x2, y2 = df.loc[0, ['x', 'y']]
        dx, dy = x2 - x1, y2 - y1
        plt.arrow(x1, y1, dx, dy, head_width=0.4, length_includes_head=True,
                  color=route_color)

plt.title("2D Store Locations and Delivery Routes by Vehicle")
plt.xlabel("X coordinate (km)")
plt.ylabel("Y coordinate (km)")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()

