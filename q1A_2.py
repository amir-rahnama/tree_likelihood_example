""" This file is created as a suggested solution template for question 1.2 in DD2434 - Assignment 1A.

    We encourage you to keep the function templates.
    However, this is not a "must" and you can code however you like.
    You can write helper functions etc. however you want.

    If you want, you can use the class structures provided to you (Node and Tree classes in Tree.py
    file), and modify them as needed. In addition to the data files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want.

    For this task, we gave you three different trees (q1A_2_small_tree, q1A_2_medium_tree, q1A_2_large_tree).
    Each tree has 5 samples (the inner nodes' values are masked with np.nan).
    We want you to calculate the likelihoods of each given sample and report it.

    Note:   The alphabet "K" is K={0,1,2,3,4}.

    Note:   A VERY COMMON MISTAKE is to use incorrect order of nodes' values in CPDs.
            theta is a list of lists, whose shape is approximately (num_nodes, K, K).
            For instance, if node "v" has a parent "u", then p(v=Zv | u=Zu) = theta[v][Zu][Zv].

            If you ever doubt your useage of theta, you can double-check this marginalization:
            \sum_{k=1}^K p(v = k | u=Zu) = 1
"""

import numpy as np
from Tree import Tree
from Tree import Node
import sys


def findPath(path, tree_topology):
    #print('path', path)
    #print(path[-1], tree_topology)

    if tree_topology[path[-1]]==-1:
        return path
    else:
        path.append(tree_topology[path[-1]])
        return findPath(path, tree_topology)

def get_prob(path, theta):
    root_prob = theta[0]
    
    #for node_idx in path:
    for i in range(len(path)):
        node_idx = path[i]
        if node_idx == 0: 
            continue
        else: 
            cpd = theta[node_idx]
            root_prob = np.dot(np.array(root_prob), np.array(cpd))
    #print(np.sum(root_prob, axis=0))
    return root_prob

def main():
    print("\n1. Load tree data from file and print it\n")

    #filename = "./data/q1A_2/q1A_2_small_tree.pkl"  
    #filename = "./data/q1A_2/q2_2_medium_tree.pkl"
        
    #t = Tree()
    #t.load_tree(filename)
   
    #print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))
    #tree_topology = t.get_topology_array()
    #theta = t.get_theta_array()
    #tree_topology = tree_topology.astype(np.int32)    
    #tree_topology[0] = -1

    filename = "./data/q1A_2/q2_2_large_tree.pkl" , 
    tree_topology = np.load('./data/q1A_2/q2_2_large_tree.pkl_topology.npy').astype(np.int32)
    tree_topology[0] = -1
    theta = np.load('./data/q1A_2/q2_2_large_tree.pkl_theta.npy', allow_pickle=True)
    
    beta_ = np.load('./data/q1A_2/q2_2_large_tree.pkl_filtered_samples.npy')
    llk = 0
    
    #This doesnt work for big tree: 
    #for sample_idx in range(t.num_samples):
    for sample_idx in range(len(beta_)):
    #    beta = t.filtered_samples[sample_idx] 
        beta = beta_[sample_idx]
        
        for i, b in enumerate(beta):
            if not np.isnan(b):
                b = int(b)
                path_ = findPath([i], tree_topology)
                
                _prob = get_prob(path_, theta)
                llk += np.log(_prob[b])
                
        print(np.exp(llk))

if __name__ == "__main__":
    main()


''' Small tree
Loading tree from  ./data/q1A_2/q1A_2_small_tree.pkl ...
K of the tree:  5       alphabet:  [0 1 2 3 4]
0.00275489358177103
2.990624667885467e-05
6.146158454393881e-07
3.2054819614983724e-09
2.1746181807538632e-11

Medium Tree
Loading tree from  ./data/q1A_2/q2_2_medium_tree.pkl ...
K of the tree:  5       alphabet:  [0 1 2 3 4]
1.1509731532264151e-18
1.9134781921027072e-36
6.576606738432932e-56
2.5250620462414495e-74
2.002245443108683e-93

Large Tree
Loading tree from  ./data/q1A_2/q2_2_large_tree.pkl ...

8.746922943362838e-74
7.82280897201882e-150
7.016555390963868e-229
3.704375613660869e-301
0.0

'''