"""
Supongamos que estamos diseñando un sistema de control de inventario para una tienda. 
Queremos optimizar el pedido de productos para mantener un nivel óptimo de inventario. Aquí está el esquema del CMDP:

1.Estados Continuos:
    El nivel de inventario actual (por ejemplo, la cantidad de productos en stock).
    El tiempo continuo (por ejemplo, días).

2.Acciones Continuas:
    La cantidad de productos que pedimos para reponer el inventario.

3.Transiciones Continuas:
    El nivel de inventario cambia según la tasa de demanda y la cantidad pedida.
    Utilizaremos una función de transición estocástica para modelar esto.

4.Recompensas Continuas:
    La recompensa está relacionada con el costo de mantener inventario y el costo de realizar pedidos.
"""


import numpy as np

# Parámetros del sistema
demand_rate = 0.1  # Tasa de demanda (productos por día)
holding_cost = 0.01  # Costo de mantener inventario por producto por día
ordering_cost = 1.0  # Costo de realizar un pedido

# Espacio de estados
inventory_levels = np.arange(0, 101)  # Niveles de inventario de 0 a 100
time_horizon = 30  # Duración del horizonte de planificación (días)

# Matriz de transición (probabilidades de transición)
def transition_probability(current_inventory, next_inventory, order_quantity):
    demand_prob = np.exp(-demand_rate * (next_inventory - current_inventory))
    order_prob = 1.0 if order_quantity == next_inventory - current_inventory else 0.0
    return demand_prob * order_prob

# Función de recompensa
def reward(current_inventory, order_quantity):
    holding_cost_total = holding_cost * current_inventory
    ordering_cost_total = ordering_cost * (order_quantity > 0)
    return -(holding_cost_total + ordering_cost_total)

# Algoritmo de programación dinámica para CMDP
V = np.zeros((len(inventory_levels), time_horizon + 1))  # Valor óptimo
policy = np.zeros((len(inventory_levels), time_horizon))  # Política óptima

for t in range(time_horizon - 1, -1, -1):
    for i, inventory in enumerate(inventory_levels):
        max_value = float("-inf")
        best_action = None
        for order in range(inventory + 1):
            value = reward(inventory, order)
            for next_inventory in inventory_levels:
                value += transition_probability(inventory, next_inventory, order) * V[next_inventory, t + 1]
            if value > max_value:
                max_value = value
                best_action = order
        V[i, t] = max_value
        policy[i, t] = best_action

print("Política óptima de pedidos:")
print(policy)
