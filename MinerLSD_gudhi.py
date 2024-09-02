import networkx as nx
import gudhi
import itertools
from graph_to_simplex import *
import random
import math
import numpy as np
from collections import Counter
from itertools import chain
# import libeval  # evaluation
# import simplex_plot # evaluation 


class MinerLSD_gudhi():

    def __init__(self):
        self.patterns = []
        self.num_lm = 0
        self.num_lme = 0


    def get_patterns(self):
        return self.patterns
    

    def get_num_lm(self):
        return self.num_lm
    

    def get_num_lme(self):
        return self.num_lme
    

    def enum(self, c, C, q, I, D, core_op, s, localmod, l, m, delta, k):
        # Increase num_lme
        self.num_lme += 1

        W_ = get_vertices(C)
        # (Optional) Check if MODL(C) > localmod before returning pattern
        # if modl_p(k, W_, C) > localmod:
        print("ENUM: ", (c, W_))
        # Store the pattern '(pattern, vertices)' in the global list
        self.patterns.append((c, W_))

        # Iterate over set of features
        for x in I - c:
            support_vert = support_of(c | {x}, delta, D)
            # G_sub = simplicial_complex_subset_vertices_to_graph(support_vert, delta)
            # W = core_op(l, support_vert, G_sub)
            W = core_op(l, k, support_vert, delta)
            if not W:
                continue
            C_new = get_subcomplex(delta, W)
            c_new = intersection_attr(W, D)

            # # if len(W) >= s and oeMODL(k, W, delta) >= localmod and c_new.isdisjoint(q):
            # subsets = split_in_l_connected_subsets(k, W, C, s)
            # for sub in subsets:
            #     C_sub = get_subcomplex(delta, sub)
            #     c_sub = intersection_attr(sub, D)
            #     if oeMODL(k, sub, delta) >= localmod and c_sub.isdisjoint(q):
            #         MinerLSD_gudhi.enum(self, c_new, C_new, q | {x}, I, D, core_op, s, localmod, k, m, delta, l)

            if len(W) >= s and is_l_connected(l, W, C_new) and oeMODL(l, W, delta) >= localmod and c_new.isdisjoint(q):
                MinerLSD_gudhi.enum(self, c_new, C_new, q, I, D, core_op, s, localmod, l, m, delta, k)
                q = q | {x}


    def miner_lsd(self, delta, I:set, D, core_op, s:float, localmod:float, l:int, k:int):
    
        self.num_lm = 0
        self.num_lme = 0

        # Get vertices of SimplicialComplex
        V = get_vertices(delta)

        # m is the number of k-simplices in delta
        m = len(get_l_simplices(l, delta))

        # Get subset of vertices
        W = core_op(l, k, V, delta)

        # libeval.save_core_size("DBLP",l,len(W))   # part of evaluation

        # Get subcomplex of SimplicialComplex containing base W
        C = get_subcomplex(delta, W)

        # Considered subset W should be larger than given bound s. 
        # # if len(W) < s or oeMODL(k, W, delta) < localmod:
        subsets = split_in_l_connected_subsets(l, W, C, s)
        for sub in subsets:
            C_sub = get_subcomplex(C, sub)
            # is_l_connected and size > s is already fulfilled by function split_in_..._subsets()
            if oeMODL(l, sub, delta) < localmod:
                continue
            else:
                MinerLSD_gudhi.enum(self, intersection_attr(sub, D), C_sub, set(), I, D, core_op, s, localmod, l, m, delta, k)

        # if len(W) < s or not is_l_connected(l, W, C) or oeMODL(l, W, delta) < localmod:
        #     return
        # else:
        #     MinerLSD_gudhi.enum(self, intersection_attr(W, D), C, set(), I, D, core_op, s, localmod, l, m, delta, k)



def l_upper_degree_bigger_than(feat, l, u, delta, D):
    """
    Add feature to vertices which have a higher l-upper-degree than u.
    Return updated feature dict. 

    Parameters:
    feat (int): feature index/number
    l (int): dimension of degree
    u (int): minimum degree to get the feature
    delta (SimplexTree)
    D (dict): Stores features of nodes
    """
    fulfilling_vertices = []
    for v in get_vertices(delta):
        if l_upper_degree(l,v,delta) > u:
            fulfilling_vertices.append(v)
    for v in fulfilling_vertices:
        D[v].add(feat)
    return D


def l_closeness_c_bigger_than_u_percentile(feat, l, u, delta, D):
    """
    Add feature to vertices which have a higher l-closeness-centrality than u.
    Return updated feature dict. 

    Parameters:
    feat (int): feature index/number
    l (int): dimension of degree
    u (int): percentile to get the feature
    delta (SimplexTree)
    D (dict): Stores features of nodes
    """

    closeness = {}

    for v in get_vertices(delta):
        val = l_closeness_centrality(l,v,delta)
        closeness.update({v: val})

    # Extract values from the dictionary
    values = list(closeness.values())

    # Calculate the 50th percentile
    percentile = np.percentile(values, u)

    # Collect vertices with closeness above the 75th percentile
    fulfilling_vertices = [v for v, value in closeness.items() if value > percentile]

    # Add closeness centrality
    for v in fulfilling_vertices:
        D[v].add(feat)
    return D


def l_distance(l, v1, v2, delta):
    """
    Returns the l-distance between vertices v1 and v2 in simplicial complex delta.

    Parameters:
    l (int): dimension of distance
    v1 (int): src. vertex index
    v2 (int): dest. vertex index
    delta (SimplexTree): simplicial complex
    """

    # check if same component
    if not is_l_connected(l, [v1, v2], delta):
        return math.inf

    # get p-adjacent vertices
    neigh_vert = l_upper_adjacent(l, v1, delta)
    if v2 in neigh_vert:
        return 1

    # (vertex_id, distance to v1)
    neigh_dist = [(vertex, 1) for vertex in neigh_vert]

    # save visisted to prevent visiting one vertex twice
    visited = {v for v in neigh_vert}
    while neigh_dist:
        vertex, dist_to_v1 = neigh_dist.pop()
        visited.add(vertex)

        # Check if v2 is reached
        if v2 == vertex:
            return dist_to_v1

        # Update distance to v1
        new_dist = dist_to_v1+1

        # Add neighbours to search
        neigh_vert = l_upper_adjacent(l,vertex,delta)
        neigh_dist += [(vertex, new_dist) for vertex in neigh_vert if vertex not in visited]
    return math.inf


def l_closeness_centrality(l, v, delta):
    """
    Returns the l-closeness-centrality for vertex v in delta.

    Parameters:
    l (int): Dimension
    v (int): vertex id
    delta (SimplexTree): Simplicial Complex
    """
    dist_sum = 0
    vertices = get_vertices(delta)
    for des in vertices:
        dist = l_distance(l, v, des, delta)
        if math.isinf(dist):
            return 0
        dist_sum += dist
    return 1 / dist_sum


def set_random_feature(delta, x, amount, D, seed=1):
    """
    delta (SimplexTree): Simplicial complex
    x (int): feature number
    amount (float): Percentage of nodes to get this feature
    D (dict): Feature dictionary
    """
    n = round(len(D)*amount)
    fulfilling_vertices = select_n_random_elements(n, delta, seed)
    for v in fulfilling_vertices:
        D[v].add(x)
    return D


def select_n_random_elements(n, delta, seed=1):
    """
    Select n random vertices from SimplexTree delta.

    Parameters:
    n (int): number of vertices to obtain the feature
    delta (SimplexTree): Simplicial complex
    """
    vertices = get_vertices(delta)
    if n > len(vertices):
        raise ValueError("n cannot be larger than the number of elements in the list.")
    
    random.seed(seed)
    return random.sample(vertices, n)


def set_feature(x, delta, D):
    match x:
        case 0:
            D = set_feature_0(delta, D)
        case 1:
            D = set_feature_1(delta, D)
        case 2:
            D = set_feature_2(delta, D)
        case 3:
            D = set_feature_3(delta, D)
        case 4:
            D = set_feature_4(delta, D)
        case 5:
            D = set_feature_5(delta, D)
        case 6:
            D = set_random_feature(delta, 3, 0.5, D)
        case 7:
            D = set_random_feature(delta, 7, 0.8, D, seed=1)
        case 8:
            D = set_random_feature(delta, 8, 0.6, D, seed=2)
        case 9:
            D = set_random_feature(delta, 9, 0.4, D, seed=3)
        case 10:
            D = set_random_feature(delta, 10, 0.2, D, seed=4)
    return D


# feature 0: 2-upper-degree > 1 for every vertex 
def set_feature_0(delta, D):
    return l_upper_degree_bigger_than(0, 2, 1, delta, D)


# feature 1: 3-upper-degree > 1 for every vertex
def set_feature_1(delta, D):
    return l_upper_degree_bigger_than(1, 3, 1, delta, D)


# feature 2: 4-upper-degree > 1 for every vertex 
def set_feature_2(delta, D):
    return l_upper_degree_bigger_than(2, 4, 1, delta, D)


# 1-Closeness-Centrality > 75th %-ile
def set_feature_3(delta, D):
    return l_closeness_c_bigger_than_u_percentile(4, 1, 75, delta, D)


# 2-Closeness-Centrality > 75th %-ile
def set_feature_4(delta, D):
    return l_closeness_c_bigger_than_u_percentile(5, 2, 75, delta, D)


# 3-Closeness-Centrality > 75th %-ile
def set_feature_5(delta, D):
    return l_closeness_c_bigger_than_u_percentile(6, 3, 75, delta, D)


def intersection_attr(W, D):
    """
    Maps a vertex subset W ⊆ V to the most specific pattern which has W as its support.
    
    Parameters:
    W (list(int)): subset of vertices to find the pattern in
    D (dict): feature dictionary to find intersection of features
    """

    # Filter W to only include keys that are in D
    valid_W = [v for v in W if v in D]
    if not valid_W:
        return set()  # Return an empty set if no valid keys are found
    return set.intersection(*[D[v] for v in valid_W])


def l_upper_degree(l, v, delta):
    """
    Counts the vertices which share the same p-simplices as vertex v.
    
    Parameters:
    l (int): Dimension of the upper (dimension) simplex.
    v (int): Vertex id to caclulate p-upper-degree of.
    delta (SimplicialComplex): From simplex.
    
    Returns:
    int: upper degree from vertex v.
    """
    shared_vertices = set()
    # Iterate through the simplices in the simplex tree
    for simplex, _ in delta.get_skeleton(l):
        # Check if the simplex has dimension p and contains the vertex v
        if len(simplex) == l + 1 and v in simplex:
            for vertex in simplex:
                if vertex != v:
                    shared_vertices.add(vertex)

    # A vertex v is p-upper-adjacent to all vertices which are vertices of the same l-simplices as v.
    return len(shared_vertices)


def simplicial_complex_subset_vertices_to_graph(W, delta):
    """
    Converts subset (defined by vertices in W) of a SimplicialComplex into a NetworkX graph.
    """
    G = nx.Graph()
    for simplex0 in W:
        G.add_node(simplex0)
        
        for (src, dest), _ in delta.get_cofaces([simplex0], 1):
            if src in W and dest in W:
                G.add_edge(src, dest)
    return G


def is_l_connected(l, W, delta_w):
    """
    True if the vertex set W of the Simplicial complex is l-connected in delta_w.
    
    Parameters:
    l (int): dimension
    W (list(int)): vertices to check if they are l-connected
    delta_w (SimplexTree): simplicial complex to check connecting l-simplices
    """

    # if a vertex v1 is in a l-connected component of a vertex v2, v1 is l-connected with v2
    component = l_connected_component(l, W[0], delta_w)
    return all(v in component for v in W)
    
    # components = []
    # for v in W:
    #     components.append(l_connected_component(k, v, delta_w))
    # return kronecker_delta(components)


def l_upper_adjacent(l, v, delta):
    """
    Return vertices that share a l-simplex with vertex v in delta.

    Parameters:
    l (int): dimension
    v (int): vertex
    delta (SimplexTree): simplicial complex
    """

    # l_cofaces = [s for s in get_l_simplices(l,delta) if v in s] # This is equivalet to get_cofaces but far slower
    # return list(set([i for vertices in l_cofaces for i in vertices if i!=v]))
    l_cofaces = delta.get_cofaces([v], l)
    return list(set([i for vertices in (i for i, _ in l_cofaces) for i in vertices if i!=v]))


def l_connected_component(l, u, delta):
    """
    Get all vertices that are p-connected to u in delta. Use BFS.

    Parameters:
    l (int): Dimension of the connecting simplices.
    u (int): Vertex index included in the component.
    delta (SimplicialComplex): u is a 0-simplex in delta.

    Returns:
    set(int): Vertex ids of p-connected component containing u.
    """
    component = {u}
    visited = []
    to_visit = {u}

    while to_visit:
        s = to_visit.pop()
        visited.append(s)
        neighbours = l_upper_adjacent(l, s, delta)
        component |= set(neighbours)
        to_visit.update([n for n in neighbours if n not in visited])

    return component


def kronecker_delta(components):
    """
    Check if all lists within a list are identical, disregarding the order of elements within the lists.

    Parameters:
    lists (list of lists of ints): The list containing sublists to be checked.

    Returns:
    bool: True if all sublists are identical, False otherwise.
    """
    if not components:
        return True  # If the list is empty, consider it as identical by definition

    first_l = components[0]
    for l in components:
        # Disregarding the order by using set()
        if set(l) != set(first_l):
            return False
    return True


def split_in_l_connected_subsets(l, W, delta_w, s):
    """
    Returns l-connected subsets of W with minimum cardinality of s.
    
    Parameters:
    l (int): dimension
    W (list(int)): vertices to split in subsets
    delta_w (SimplexTree): simplicial complex to access l-simplices
    s (int): minimum number of vertices in a subset
    """
    
    # Vertices with the same l-connected component are l-connected
    components = {}
    for v in W:
        components.update({v : l_connected_component(l, v, delta_w)})

    if not components:
        return []

    value_to_keys = {}

    for key, value_set in components.items():
        # Convert the set to a frozenset so it can be used as a key in the dictionary
        frozen_value_set = frozenset(value_set)

        # Group the keys by their corresponding value set
        if frozen_value_set in value_to_keys:
            value_to_keys[frozen_value_set].append(key)
        else:
            value_to_keys[frozen_value_set] = [key]

    # Convert the grouped keys into a list of lists
    subsets = list(value_to_keys.values())
    return [sub for sub in subsets if len(sub) >= s]


def modl_l(l, W, delta):
    """
    l-local Modularity.

    Parameters:
    l (int): dimension
    W (list(int)): vertices
    delta_w (SimplexTree): simplicial complex to access l-simplices
    """

    m = len(get_l_simplices(l, delta))
    delta_w = get_subcomplex(delta, W)
    l_simplices = get_l_simplices(l, delta_w)
    m_w = len(l_simplices)
    modl_sum = 0

    # Iterate over all combinations of p+1 vertices from W
    for combination in itertools.combinations(W, l+1):
        degree_product = 1
        connected_components =[]
        
        for vertex in combination:
            degree_product *= l_upper_degree(l, vertex, delta)

            # Get p-connected components
            connected_components.append(l_connected_component(l, vertex, delta))
        
        # Apply the condition function
        delta_value = kronecker_delta(connected_components)

        # Add to the sum
        modl_sum += degree_product * delta_value
        
    # Normalize the sum
    normalization_factor = ((l+1)**(l+1))*(m**(l+1))
    if normalization_factor != 0:
        modl_score = (m_w / m) - (modl_sum / normalization_factor)
    else:
        modl_score = 0

    return modl_score


def oeMODL(l, W, delta):
    """
    Calculates the optimistic estimate of the l-local Modularity of a subset W of a SimplicialComplex.
    
    Parameters:
    l (int): dimension
    W (list(int)): vertices
    delta_w (SimplexTree): simplicial complex to access l-simplices
    """
    # m is the number of l-simplices in delta
    m = len(get_l_simplices(l, delta))
    delta_w = get_subcomplex(delta, W)
    # m_w denotes the number of l-simplices in the induced subcomplex delta_w of delta
    m_w = len(get_l_simplices(l, delta_w))

    if m_w >= (((l+1)**(l-1))*(m**(l)))/2:
        return ((l+1)**(l-1))*(m**(l-1))
    else:
        return (m_w/m)-(m_w**2)/(((l+1)**(l-1))*(m**(l+1)))


def find_cliques_of_size(G, size):
    """
    Find all cliques of a specific size in the graph.

    Parameters:
    G (nx.Graph): The input graph.
    size (int): The size of the cliques to find.

    Returns:
    list: A list of cliques, where each clique is a list of nodes.
    """
    return [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == size]


def get_subcomplex(delta, W):
    """
    Get the Subcomplex of delta containing the vertices W.
    
    Parameters:
    delta (SimplexTree): The input simplicial complex.
    W (list(int)): All vertices of the subset.

    Returns:
    Simplicial Complex: Subset of delta.
    """

    simplices = [simplex[0] for simplex in delta.get_simplices()]
    c = gudhi.SimplexTree()
    for s in simplices:
        # Copy simplices if all vertices of simplex are in W
        if all(element in W for element in s):
            c.insert(s)
    return c


def support_of(c, delta, D):
    """
    Returns all vertices that have the feature c.

    Parameters:
    c (list(int)): Pattern to get the support of.
    delta (SimplexTree): to search the pattern in it
    D (dict): Dataset which holds the features of each vertex.
    """

    vertices = get_vertices(delta)
    return {v for v in vertices if v in D and c.issubset(D[v])}


def l_simplex_degree(l, delta):
    """
    Return a dictionary with vertices of SimplexTree delta as keys and
    l-simplex-degree of the vertices as values. The l-simplex-degree of 
    a vertex is the number ob l-simplices where the vertex is in.

    Parameters:
    l (int): dimension
    delta (SimplexTree): simplicial complex
    """

    l_simplices = get_l_simplices(l,delta)
    vert_count = Counter(chain.from_iterable(set(x) for x in l_simplices))
    vertices = get_vertices(delta)
    degrees = {}
    for v in vertices:
        degrees.update({v : vert_count[v]})
    return degrees


def l_core_number(l, delta):
    """
    Returns a dict with vertices as keys and the value of a vertex is 
    the l-core number (the maximum core with l-simplices).
    For example: {0: k} -> vertex 0 is part of k-l-core, 
    while every vertex in the l-k-core has minimum k l-simplices.
    (Important: All simplices are considered where all vertices are completly 
    in the core.)

    Parameters:
    l (int): simplex dimension
    delta (SimplexTree): is already only consisting of vertices V here.
    """

    vertices = get_vertices(delta)
    degrees = l_simplex_degree(l, delta)
    # Sort nodes by degree
    nodes = sorted(degrees, key=degrees.get)
    # The initial guess for the core number of a node is its degree.
    core = degrees.copy()
    
    nbrs = {v: list(l_upper_adjacent(l, v, delta)) for v in vertices}
    nbrs_simpl = {v: list(adjacent_l_simplices(l, v, delta)) for v in vertices}
    while nodes:
        # Find the element with the lowest value in the dictionary
        v = nodes[0]
        if len(nodes) > 1:
            for n in nodes[1:]:
                if core[n] < core[v]:
                    v = n
        
        # Pop the element from the list
        nodes.remove(v)

        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)

                # Remove all simplices that contain v, since v has lower core number
                size_simpl = len(nbrs_simpl[u])
                nbrs_simpl[u] = [s for s in nbrs_simpl[u] if v not in s]
                size_simpl_after = len(nbrs_simpl[u])
                num_del_smpl = size_simpl-size_simpl_after

                # Update core number
                core[u] -= num_del_smpl
    return core


# def l_core_number(l, delta):
#     """
#     Returns a dict with vertices as keys and the value of a vertex is 
#     the l-core number (the maximum core with l-simplices).
#     For example: {0: k} -> vertex 0 is part of k-l-core, 
#     while every vertex in the p-l-core has minimum k l-simplices.
#     (Important: All simplices are considered where all vertices are completly 
#     in the core.)

#     l (int): simplex dimension
#     delta (SimplexTree): is already only consisting of vertices V here.
#     """

#     vertices = get_vertices(delta)
#     degrees = l_simplex_degree(l, delta)
#     # degrees = dict([(v, l_upper_degree(l, v, delta)) for v in vertices])
#     # Sort nodes by degree
#     nodes = sorted(degrees, key=degrees.get)
#     bin_boundaries = [0]
#     curr_degree = 0
#     for i,v in enumerate(nodes):
#         if degrees[v] > curr_degree:
#             bin_boundaries.extend([i] * (degrees[v] - curr_degree))
#             curr_degree = degrees[v]
#     node_pos = {v: pos for pos, v in enumerate(nodes)}
#     # The initial guess for the core number of a node is its degree.
#     core = degrees.copy()
#     nbrs = {v: list(l_upper_adjacent(l, v, delta)) for v in vertices}
#     nbrs_simpl = {v: list(adjacent_l_simplices(l, v, delta)) for v in vertices}
#     for v in nodes:
        
#         for u in nbrs[v]:
#             if core[u] > core[v]:
#                 nbrs[u].remove(v)

#                 # Remove all simplices that contain v, since v has lower core number
#                 size_simpl = len(nbrs_simpl[u])
#                 nbrs_simpl[u] = [s for s in nbrs_simpl[u] if v not in s]
#                 size_simpl_after = len(nbrs_simpl[u])
#                 num_del_smpl = size_simpl-size_simpl_after

#                 pos = node_pos[u]
#                 bin_start = bin_boundaries[core[u]]
#                 node_pos[u] = bin_start
#                 node_pos[nodes[bin_start]] = pos
#                 nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
#                 bin_boundaries[core[u]] += 1

#                 # if num_del_smpl > 1:
#                 # Update core number
#                 core[u] -= 1
#     return core


def _core_subcomplex(l, delta, k_filter, k=None, core=None):
    if core is None:
        core = l_core_number(l, delta)
    if k is None:
        k = max(core.values())
    nodes = (v for v in core if k_filter(v, k, core))
    return nodes
    # return get_subcomplex(delta)


def adjacent_l_simplices(l, v, delta):
    """
    Returns l-simplices where v is a vertex.

    Parameters:
    l (int): dimension
    v (int): vertex
    delta (SimplexTree): simplicial complex
    """

    l_cofaces_w_filtration = delta.get_cofaces([v], l)
    l_cofaces = (n for n, _ in l_cofaces_w_filtration)
    return [v for v in l_cofaces]


def k_l_core(l, k, V, delta, l_core_number=None):
    """
    Every vertex in the (k,l)-core is a vertex of minimum k l-simplices.

    Parameters:
    l (int): dimension
    k (int): number of l-simplices per vertex
    V (list(int)): subset of vertices to find (k,l)-core
    l_core_number (dict): l-core number per vertex

    Returns: (k,l)-core
    """
    delta_v = get_subcomplex(delta, V)

    def k_filter(v, k, c):
        return c[v] >= k
    nodes = _core_subcomplex(l, delta_v, k_filter, k, l_core_number)
    return [n for n in nodes]


def l_core_number_vertices(l, delta):
    """
    Returns dict with vertices as keys and values as number of vertices
    that are l_upper_adjacent (meaning that the vertices are connected by one l-simplex from the vertex).
    
    Parameters:
    l (int): dimension
    delta (SimplexTree): only consists of vertices V here
    """

    vertices = get_vertices(delta)
    degrees = dict([(v, l_upper_degree(l, v, delta)) for v in vertices])
    # Sort nodes by degree
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i,v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(l_upper_adjacent(l, v, delta)) for v in vertices}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


########### Example usage ###########
# Reconstruct Attributed Network from MinerLC Example (https://lipn.univ-paris13.fr/MinerLC/)

# Build a graph
vertices = [0, 1, 2, 3, 4, 5, 6, 7]
edges = [
    [0, 1], [0, 2], [1, 2], [2, 3], [3, 4], 
    [4, 5], [5, 6], [6, 7], [5, 7], [4, 6]
]
G = nx.Graph()
G.add_edges_from(edges)

# Transform graph to simplicial complex
delta = graph_to_simplicial_complex(G)

# Check if it worked
print("Graph Nodes:", G.nodes())
print("Graph Edges:", G.edges())
print("Simplices:", [s for s in delta.get_simplices()])

# Define a set of features I
I = {0, 1, 2}

# Define a dataset D to assign features to vertices
D = {0: {0, 1}, 1: {0, 1}, 2: {0, 1}, 3: {0, 2}, 4: {0, 2}, 5: {1, 2}, 6: {0, 1, 2}, 7: {1, 2}}

# Set threshold s and localmod
s = 2
localmod = 0.1 

# Set dimension of simplices to be observed
l = 2

# Set amount of l-simplices needed per vertex to be in the (k,l)-core
k = 1

# Run the MINERLSD function
mLSD = MinerLSD_gudhi()
mLSD.miner_lsd(delta, I, D, k_l_core, s, localmod, l, k)

# Display results
print("Patterns found:" ,mLSD.patterns)
# simplex_plot.highlight_patterns_nx2(G, mLSD.patterns, D)