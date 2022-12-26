#use particle swarm to solve max flow
import numpy as np

def pso(func, bounds, num_particles, max_iter,w = 0.7,c1 = 1.5,c2 = 1.5):
    dimensions = len(bounds)
    
    # Initialize the particles
    particles = []
    for i in range(num_particles):
        particle = {
            'position': [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)],
            'velocity': [0 for _ in range(dimensions)],
            'best_position': None,
            'best_value': float('inf')
        }
        particles.append(particle)
    
    # Set the global best position and value to infinity
    global_best_position = [float('inf') for _ in range(dimensions)]
    global_best_value = float('inf')
    
    
    # Run the optimization loop
    for i in range(max_iter):
        for particle in particles:
            # Evaluate the function at the particle's current position
            value = func(particle['position'])
            
            # Update the particle's best position and value if necessary
            if value < particle['best_value']:
                particle['best_position'] = particle['position'][:]
                particle['best_value'] = value
            
            # Update the global best position and value if necessary
            if value < global_best_value:
                global_best_position = particle['position'][:]
                global_best_value = value
            
            # Update the particle's velocity and position
            for j in range(dimensions):
                particle['velocity'][j] = w * particle['velocity'][j] \
                    + c1 * np.random.uniform(0, 1) * (particle['best_position'][j] - particle['position'][j]) \
                    + c2 * np.random.uniform(0, 1) * (global_best_position[j] - particle['position'][j])
                particle['position'][j] += particle['velocity'][j]
    
    # Return the global best position and value
    return global_best_position, global_best_value

def quadratic(x):
    return x[0] ** 2 + x[1] ** 2

def aco_max_flow(graph, num_ants, max_iter, pheromone_evaporation_rate, pheromone_deposit_rate):
    # Get the number of nodes in the graph
    num_nodes = len(graph)
    
    # Initialize the pheromone matrix
    pheromone = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    # Define the fitness function
    def fitness(position):
        sink, source = position[0], position[1]
        max_flow = 0
        # Compute the maximum flow from the source to the sink
        residual_graph = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                residual_graph[i][j] = graph[i][j]
        
        # Define a list to store the parent nodes in the breadth-first search
        parent = [-1 for _ in range(num_nodes)]
        # using the graph and the given sink and source nodes
        # (you can use any maximum flow algorithm here, such as the Ford-Fulkerson
        while True:
            # Perform a breadth-first search to find a path with available capacity
            q = []
            visited = [False for _ in range(num_nodes)]
            q.append(source)
            visited[source] = True
            while q:
                u = q.pop(0)
                for v in range(num_nodes):
                    if not visited[v] and residual_graph[u][v] > 0:
                        q.append(v)
                        visited[v] = True
                        parent[v] = u
            # If we couldn't find a path with available capacity, break out of the loop
            if not visited[sink]:
                break
            
            # Find the minimum capacity along the path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, residual_graph[parent[s]][s])
                s = parent[s]
            
            # Update the residual capacities of the path and reverse edges
            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                residual_graph[u][v] -= path_flow
                residual_graph[v][u] += path_flow
                v = parent[v]
        
        # Return the maximum flow
        return max_flow
    
    # Initialize the best solution
    best_solution = None
    best_fitness = float('-inf')
    
    # Run the ACO algorithm for the specified number of iterations
    for i in range(max_iter):
        # Initialize the solutions for each ant
        solutions = []
        
        # Run the ACO algorithm for each ant
        for j in range(num_ants):
            # Initialize the current solution
            current_solution = []
            
            # Choose a random starting node for the ant
            current_node = np.random.randint(num_nodes)
            current_solution.append(current_node)
            
            # Move the ant to a new node according to the pheromone levels and the graph structure
            while True:
                next_node = None
                max_prob = 0
                for k in range(num_nodes):
                    if graph[current_node][k] > 0:  # check if there is an edge between the current node and node k
                        prob = (pheromone[current_node][k] ** pheromone_deposit_rate) * ((1.0 / graph[current_node][k]) ** pheromone_evaporation_rate)
                        if prob > max_prob:
                            max_prob = prob
                            next_node = k
                if next_node is None:
                    break
                current_solution.append(next_node)
                current_node = next_node
            
            # Evaluate the fitness of the solution and add it to the list of solutions
            fitness_value = fitness(current_solution)
            solutions.append((current_solution, fitness_value))
            
            if fitness_value > best_fitness:
                best_solution = current_solution
                best_fitness = fitness_value
        
        # Update the pheromone levels on the graph according to the solutions found by the ants
        for solution, fitness_value in solutions:
            for i in range(num_nodes - 1):
                pheromone[solution[i]][solution[i+1]] += fitness_value
        
        # Evaporate the pheromone levels on the graph
        for i in range(num_nodes):
            for j in range(num_nodes):
                pheromone[i][j] *= (1 - pheromone_evaporation_rate)
    
    # Return the best solution found by the ACO algorithm
    return best_solution       
if __name__ == '__main__':
    print(pso(quadratic, [(-10, 10), (-10, 10)], 10, 1000))
    #print(quadratic([-3.410288654851112e-09, 6.273175905378035]))