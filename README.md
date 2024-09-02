# minerlsd_simplicial

## Abstract
Local pattern mining on attributed graphs aims to detect subsets of vertices that are induced by patterns composed of specific sets of attributes. This paper generalizes such graph-based approaches to simplicial complexes building on the MINERLSD algorithm for efficient local pattern mining. We present according generalizations of the graph-based closed pattern case to simplicial complexes, as well as a generalization of the Modularity for simplicial complexes in order to enable an efficient pattern mining approach. We demonstrate the efficacy of the proposed approach via experimentation using several datasets.

## Overview
MinerLSD_gudhi.py implements the MinerLSD (simplicial version) with a demonstrative example.
graph_to_simplex.py is a library that creates a simplicial complex out of a graph.
