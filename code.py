import numpy as np
seed = 1
np.random.seed(seed) #for repeatability
import networkx as nx
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
def search_neighbors(graph, node):
  '''
  This function will search for their neighbors as a list of tuples (node,edge)
  '''
  neighbors = []
  for i in graph[node]:
    neighbors.append((i))
  return neighbors
def search_edges(graph, node):
  '''
  This function will search for their neighbors as a list of tuples (origin,destination)
  '''
  neighbors = []
  for i in graph[node]:
    neighbors.append((node,i))
  return neighbors
def find_side_edges(graph, node,sides):
  '''
  This function will search for their neighbors as a list of tuples (origin,destination)
  '''
  neighbors = []
  for i in list(graph.in_edges(node)):
    if graph[i[0]][i[1]]["sides"] == sides:
      neighbors.append((i[0],i[1]))
  return neighbors
def spread_length(graph,side):
  '''
  This function will return the number of edges that pick the side side
  '''
  count = 0
  for i in graph.edges():
    if graph[i[0]][i[1]]["sides"] == side:
      count += 1
  return count
def color_Graph(graph):
  '''
  This function will color the graph based on the sides
  '''
  color_map = []
  for i in graph.nodes:
    if graph.nodes[i]['sides'] == 0:
      color_map.append('gray')
    elif graph.nodes[i]['sides'] == 1:
      color_map.append('blue')
    else:
      color_map.append('red')
  return color_map
def color_edge(graph):
    '''
    This function will color the graph based on the sides
    '''
    color_map = []
    for i in graph.edges:
      if graph.edges[i]['sides'] == 0:
        color_map.append('black')
      elif graph.edges[i]['sides'] == 1:
        color_map.append('blue')
      else:
        color_map.append('red')
    return color_map
def plot_graph(graph,**kwargs):
  '''
  This function will plot the graph
  '''
  try:
    spring = kwargs['spring']
    circular = kwargs['circular']
  except:
    spring = False
    circular = False
  if spring:
    pos = nx.spring_layout(graph,scale=2)
    y = nx.draw(graph, node_color=color_Graph(graph),edge_color=color_edge(graph),with_labels=True, pos=pos)
  elif circular:
    pos = nx.circular_layout(graph)
    y = nx.draw(graph, node_color=color_Graph(graph),edge_color=color_edge(graph),with_labels=True, pos=pos)
  else:
    pos = nx.shell_layout(graph,scale=2)
    y = nx.draw(graph, node_color=color_Graph(graph),edge_color=color_edge(graph),with_labels=True, pos=pos)
  return y
def init_graph2(n=20,p=0.1, prob_p = None):
  directed = True #determine it is a directed graph or not
  G = nx.erdos_renyi_graph(n,p,directed=directed)
  if prob_p is None:
    prob_prop = np.random.uniform(0,1)
  else:
    prob_prop = prob_p
  sides = []
  nx.set_node_attributes(G, sides,"sides")
  nx.set_node_attributes(G, prob_prop,"prob_prop")
  for i in G.nodes:
    G.nodes[i]["sides"] = 0
  features = {}
  sides = np.zeros(n)
  done = np.zeros(n)
  sides = []
  prob_propA = []
  prob_propB = []
  nx.set_edge_attributes(G, sides, "sides")
  nx.set_edge_attributes(G, prob_propA, "Aprop")
  nx.set_edge_attributes(G, prob_propB, "Bprop")
  for i in G.edges():
    G[i[0]][i[1]]["sides"] = 0
    G[i[0]][i[1]]["Aprop"] = prob_prop
    G[i[0]][i[1]]["Bprop"] = prob_prop
  return G
def find_sides_edges(graph,node,side):
    '''
    This function will find the edges that point to the node with the target side
    '''
    edges = []
    for i in graph.in_edges(node):
        if graph[i[0]][i[1]]["sides"] == side:
            edges.append(i)
    return edges
def choose_sides(edges_from_A, edges_from_B,graph,node,prob_prop,init = False):
    '''
    This function will choose the sides for the edges
    '''
    nA = len(edges_from_A)
    nB = len(edges_from_B)
    denom = nA + nB + 0.0001
    for i in edges_from_A:
        graph[i[0]][i[1]]["Aprop"] *=  prob_prop
        #print(graph[i[0]][i[1]]["Aprop"])
    prob_propA = sum(graph[i[0]][i[1]]["Aprop"] for i in edges_from_A)/denom
    for i in edges_from_B:
        graph[i[0]][i[1]]["Bprop"] *=  prob_prop
        #print(graph[i[0]][i[1]]["Bprop"])
    prob_propB = sum(graph[i[0]][i[1]]["Bprop"] for i in edges_from_B)/denom
    if init:
        if graph.nodes[node]["sides"] == 1:
            prob_propA = 1
            prob_propB = 0
        else:
            prob_propA = 0
            prob_propB = 1
    #partition [0,1] into three parts: A, B, and the rest
    #A: [0,prob_propA]
    #B: [prob_propA, prob_propA + prob_propB]
    #rest: [prob_propA + prob_propB, 1]
    choice = []
    for i in graph.out_edges(node):
        r = np.random.uniform(0,1)
        if r < prob_propA:
            choice.append(1)
        elif r < prob_propB + prob_propA:
            choice.append(2)
        else:
            choice.append(0)
    #print(f"from node {node} :prob_propA = {prob_propA}, prob_propB = {prob_propB}")
    #print(f"choice = {choice}")
    return prob_propA, prob_propB, choice
def cascade_edge(graph=init_graph2(), init_1 = None, init_2 = None):
    '''
    This function will perform the information cascading on edges
    '''
    if init_1 == None:
        init_1 = np.random.choice(graph.nodes)
    if init_2 == None:
        while True:
            init_2 = np.random.choice(graph.nodes)
            if init_2 != init_1:
                break
    prob_prop = graph.nodes[init_1]["prob_prop"]
    graph.nodes[init_1]["sides"] = 1
    graph.nodes[init_2]["sides"] = 2
    #set probability of propagation of initial nodes to be 1
    for i in graph.out_edges(init_1):
        graph[i[0]][i[1]]["Aprop"] = 1
    for i in graph.out_edges(init_2):
        graph[i[0]][i[1]]["Bprop"] = 1
    #set all edges from init_1 to their neighbors to side 1
    for i in graph.neighbors(init_1):
        graph[init_1][i]["sides"] = 1
    #set all edges from init_2 to their neighbors to side 2
    for i in graph.neighbors(init_2):
        graph[init_2][i]["sides"] = 2
    #now we do iterative deepening to explore the graph from both sides adversarially
    #coin toss choosing the side to explore first
    if np.random.random() < 0.5:
        first_explore = init_1
        second_explore = init_2
    else:
        first_explore = init_2
        second_explore = init_1
    #now we do iterative deepening from first explore alternate with second explore
    #we will stop when we reach a node that has both sides
    queue = [first_explore,second_explore]
    already_explored = []
    init = True
    while len(queue) > 0:
        node = queue.pop(0)
        #print(f"node = {node}")
        #find the edges that point to the node with the target side
        edges_from_A = find_side_edges(graph,node,1)
        #print(f"edges_from_A = {edges_from_A}")
        edges_from_B = find_side_edges(graph,node,2)
        #print(f"edges_from_B = {edges_from_B}")
        #choose the sides for the edges
        prob_propA, prob_propB, choice = choose_sides(edges_from_A, edges_from_B,graph,node,prob_prop,init = init)
        for i in range(len(graph.out_edges(node))):
            if choice[i] == 1:
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["sides"] = 1
                #print("edge from",list(graph.out_edges(node))[i][0],"to",list(graph.out_edges(node))[i][1],"is side 1")
                #now set the probability of propagation of the edge to be prob_propA
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Aprop"] = prob_propA
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Bprop"] = prob_propB
                if list(graph.out_edges(node))[i][1] not in queue and list(graph.out_edges(node))[i][1] not in already_explored:
                    queue.append(list(graph.out_edges(node))[i][1])
            elif choice[i] == 2:
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["sides"] = 2
                #now set the probability of propagation of the edge to be prob_propA
                #print("edge from",list(graph.out_edges(node))[i][0],"to",list(graph.out_edges(node))[i][1],"is side 2")
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Aprop"] = prob_propA
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Bprop"] = prob_propB
                if list(graph.out_edges(node))[i][1] not in queue and list(graph.out_edges(node))[i][1] not in already_explored:
                    queue.append(list(graph.out_edges(node))[i][1])
            else:
                #now set the probability of propagation of the edge to be prob_propA
                #print("edge from",list(graph.out_edges(node))[i][0],"to",list(graph.out_edges(node))[i][1],"is side 0")
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Aprop"] = 0
                graph[list(graph.out_edges(node))[i][0]][list(graph.out_edges(node))[i][1]]["Bprop"] = 0
            #print(queue)
            if second_explore in already_explored:
                init = False
            already_explored.append(node)
   # print("spread length of A:",spread_length(graph,1))
   # print("spread length of A:",spread_length(graph,2))
    return graph, spread_length(graph,1)/graph.number_of_edges(), spread_length(graph,2)/graph.number_of_edges()
#naive apporach: max flow
def max_flow_leaks(graph,source,sink):
    '''
    this function will calculate the max flow with leaks on the graph
    '''
    #first, we need to create a residual graph
    residual_graph = nx.DiGraph()
    for edge in graph.edges:
        residual_graph.add_edge(edge[0],edge[1],capacity=graph[edge[0]][edge[1]]["Aprop"])
        residual_graph.add_edge(edge[0],edge[1],capacity=graph[edge[0]][edge[1]]["Bprop"])
    #now we need to find the max flow
    max_flow = nx.maximum_flow_value(residual_graph,source,sink)
    return max_flow
def max_flow_search(graph):
    '''
    this function will return the source that will maximize the max flow
    '''
    n = int(graph.number_of_nodes())
    flow = []
    for i in tqdm(range(n)):
        fx = []
        for j in range(n):
            try:
                fx.append(max_flow_leaks(graph,i,j))
            except:
                fx.append(0)
        flow.append(fx)
    return np.argmax(np.mean(flow,axis=1))
def filter_attrib_matrix(attrib_matrix, side):
    '''
    This function will filter the attribute matrix based on the side
    '''
    x = np.zeros((attrib_matrix.shape[0],attrib_matrix.shape[1]))
    for i in range(attrib_matrix.shape[0]):
        for j in range(attrib_matrix.shape[1]):
            if attrib_matrix[i,j] == side:
                x[i,j] = 1
    return x
def create_density(graph,init_1,init_2, n = 100):
    Den1,Den2 = [],[]
    for i in range(n):
        new_graph = graph.copy()
        temp,_,_ = cascade_edge(new_graph,init_1,init_2)
        mm = nx.attr_matrix(temp,'sides',rc_order=list(range(temp.number_of_nodes())))
        x = filter_attrib_matrix(mm,1)
        y = filter_attrib_matrix(mm,2)
        Den1.append(x)
        Den2.append(y)
    den1 = np.mean(Den1,axis=0)
    den2 = np.mean(Den2,axis=0)
    cutting = np.zeros((den1.shape[0],den1.shape[1]))
    for i in range(den1.shape[0]):
        for j in range(den1.shape[1]):
            if den1[i,j] < 0.5 and den1[i,j] != 0 and den2[i,j]!=0:
                cutting[i,j] = 1
    return den1,den2, cutting
def pruning_graph(cutting,graph):
    '''
    This function will prune the graph based on the cutting matrix
    '''
    new_graph = graph.copy()
    for edge in graph.edges:
        if cutting[edge[0],edge[1]] == 1:
            new_graph.remove_edge(edge[0],edge[1])
    return new_graph
def cutting_generator(G):
    Den1, Den2 = [],[]
    for i in range(G.number_of_nodes()):
        den1_s, den2_s = [],[]
        for j in range(G.number_of_nodes()):
            #create density tensor to find the maximum likelihood of the edge
            # to be cut
            den1,den2,_ = create_density(G,i,j)
            den1_s.append(den1)
            den2_s.append(den2)
        Den1.append(den1_s)
        Den2.append(den2_s)
    #find the maximum likelihood of the edge to be cut
    Den1 = np.array(Den1)
    Den2 = np.array(Den2)
    k = np.zeros((Den1.shape[0],Den1.shape[1]))
    p = np.mean(Den1,axis = (2,3)) - np.mean(Den2,axis = (2,3))
    for i in range(Den1.shape[0]):
        for j in range(Den1.shape[1]):
            k[i,j] = 1 if p[i,j] < 0 and (i,j) in G.edges else 0
    return k
def find_optimal_solution(graph):
    '''
    This function will find the optimal solution for the game
    '''
    n = graph.number_of_nodes()
    maxj,value = 0,0
    for i in range(n):
        for j in range(n):
            try:
                Gp,a,b = cascade_edge(graph,i,j)
                if a > value:
                    maxj = j
                    value = a
            except:
                pass
    return maxj,value
def main():
    aa,bb = [],[]
    maxis, values = [],[]
    for i in tqdm(range(10)):
        Grap = init_graph2(30,0.2,0.5)
        new_graph2 = Grap.copy()
        k = cutting_generator(Grap)
        new_graph3 = pruning_graph(k,new_graph2)
        start = max_flow_search(new_graph3)
        maxj,value = 0,0
        for j in range(Grap.number_of_nodes()):
            aaa, bbb = [],[]
            try:
                Gp,a,b = cascade_edge(new_graph2,start,j)
                aaa.append(a)
                bbb.append(b)
            except:
                aaa.append(0)
                bbb.append(0)
        aa.append(aaa)
        bb.append(bbb)
        maxi,value = find_optimal_solution(new_graph2)
        maxis.append(maxi)
        values.append(value)
    print(f"average win of A by cutting method: {np.mean(aa)}")
    print(f"average win of B by cutting method: {np.mean(bb)}")
    #random strategy
    ex2a, ex2b = [],[]
    for i in tqdm(range(10)):
        Grap = init_graph2(30,0.2,0.5)
        new_graph2 = Grap.copy()
        try:
            Gp,a,b = cascade_edge(new_graph2)
            ex2a.append(float(a))
            ex2b.append(float(b))
        except:
            ex2a.append(0)
            ex2b.append(0)

    #whisker-box plot
    plt.boxplot([ex2a,ex2b,x,y],labels=['A Random','B Random','A Cutting','B Cutting'])
    #set a title
    plt.title('Comparison of the probability of winning in each side\n from Random and Cutting method on a fixed graph')
    plt.savefig('boxplot2.pdf',dpi=1200)
if __name__ == '__main__':
    main()