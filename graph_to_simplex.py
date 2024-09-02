import gudhi
import networkx as nx
import itertools


def get_vertices(simplex_tree):
    """
    Return all 0-simplices (vertices) from simplex_tree.
    """
    # simplex is a tuple containing vertex list and filtration value
    return [simplex[0][0] for simplex in simplex_tree.get_skeleton(0)]


def get_l_simplices(l, simplex_tree):
    """
    Return all l-simplices from simplex_tree.
    """
    return [simplex[0] for simplex in simplex_tree.get_skeleton(l) if len(simplex[0])==l+1]


def convert_node_names_to_integers(G):
    """
    Convert node names in the graph G and keys in the dataset to integers ranging from 0 to number_of_nodes - 1.

    Parameters:
    G (networkx.Graph): The input graph with any type of node names.

    Returns:
    networkx.Graph, dict: A new graph with node names as integers and a new dataset with integer keys.
    """
    # Create a mapping from old node names to new integer labels
    mapping = {old_name: new_name for new_name, old_name in enumerate(G.nodes())}
    
    # Relabel the graph nodes using the mapping
    G_int = nx.relabel_nodes(G, mapping)
    
    return mapping, G_int


def convert_node_names_to_integers_in_D(mapping, dataset):
    # Relabel the dataset keys using the same mapping
    dataset_int = {mapping[old_key]: value for old_key, value in dataset.items()}
    return dataset_int


def create_simplex_tree(vertices, edges, higher_dim_simplices):
    """
    Create SimplexTree from vertices, edges and higher-order simplices.

    Parameters:
    vertices (list(int))
    edges (list(tuples(int)))
    higher_dim_simplices(list(list(int)))
    """
    st = gudhi.SimplexTree()
    for v in vertices:
        st.insert([v])
    for e in edges:
        st.insert(list(e))
    for h in higher_dim_simplices:
        st.insert(list(h))
    return st


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


def graph_to_simplicial_complex(G):
    """Convert Graph G to simplicial complex."""
    s3 = find_cliques_of_size(G, 3)
    s4 = find_cliques_of_size(G, 4)
    s5 = find_cliques_of_size(G, 5)
    s6 = find_cliques_of_size(G, 6)
    s7 = find_cliques_of_size(G, 7)

    higher_dim_simplices = list(itertools.chain(s3,s4,s5,s6,s7))
    return create_simplex_tree(G.nodes(), G.edges(), higher_dim_simplices)


def simplex_tree_to_graph(simplex_tree):
    """Converts a Simplex Tree into a NetworkX graph."""
    G = nx.Graph()
    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 1:
            G.add_node(list(simplex[0])[0])
        elif len(simplex[0]) == 2:
            G.add_edge(*simplex[0])
    return G