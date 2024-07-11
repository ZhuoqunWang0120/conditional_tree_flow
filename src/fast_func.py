import treenode
import copy
import numpy as np

def model2prob(tree: treenode.TreeNode, X: np.ndarray) -> np.ndarray:
    o=[]
    tree.getattr_bfs('params',o)
    model = [x['model'] if x else None for x in o]
    theta = np.array([x['theta'] for i,x in enumerate(o) if x and x['theta'] and model[i] and model[i]!='one-side'])
    model = np.array(model)
    pleft = []
    tree.getattr_bfs('pleft', pleft)
    pleft = np.array(pleft)
    P = np.array([pleft for _ in range(X.shape[0])], dtype = float) # nrow = nsample
    idx = model.astype(bool) & (model!='one-side')
    if np.any(idx):
        p1 = np.array([m.predict_proba(X)[:,0] for m in theta])
        # P = np.array([pleft for _ in range(p1.shape[1])], dtype = float) # nrow = nsample
        P[:,idx] = p1.T
    return P

def scale_dependent_shrinkage_helper(tree: treenode.KeyTreeNode, c0: float, gamma:float, p_output: np.ndarray):
    if not tree or not tree.left or not tree.right:
        pass
    else:
        mu_left = np.prod(tree.left.right_bound - tree.left.left_bound)
        mu_A = np.prod(tree.right_bound - tree.left_bound)
        vol_A = mu_A 
        c_A = c0 * (1 - np.log2(vol_A)) ** ( - gamma )
        cond_mu_left = mu_left / mu_A
        key = int(tree.key)
        p_output[:,key] = (1 - c_A) * cond_mu_left + c_A * p_output[:,key]
        scale_dependent_shrinkage_helper(tree.left, c0, gamma, p_output)
        scale_dependent_shrinkage_helper(tree.right, c0, gamma, p_output)

def scale_dependent_shrinkage(tree: treenode.KeyTreeNode, c0: float, gamma:float, pleft: np.ndarray) -> np.ndarray:
    p_output = pleft.copy()
    scale_dependent_shrinkage_helper(tree, c0, gamma, p_output)
    return p_output
def treecdf_A(tree: treenode.KeyTreeNode, pleft: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    tree: shared tree structure
    pleft: nsample * nnode, bfs
    residual: nsample * d
    """
    if not tree.left or not tree.right: # leaf
            return residual
    else:
        key = int(tree.key)
        x_copy = residual.copy()
        k = tree.split_axis
        mu_left = (tree.split_point - tree.left_bound[k]) / (tree.right_bound[k] - tree.left_bound[k])
        left_idx = x_copy[:,k]<=tree.split_point
        right_idx = x_copy[:,k]>tree.split_point
        # left child
        x_copy[left_idx, k] = pleft[left_idx, key] / mu_left * (x_copy[left_idx, k] - tree.left_bound[k]) + tree.left_bound[k]
        x_copy[x_copy[:,k] == tree.left_bound[k], k] = np.nextafter(tree.left_bound[k], np.inf)
        # right child
        mu_right = 1 - mu_left
        x_copy[right_idx, k] = (1 - pleft[right_idx, key])/mu_right * (x_copy[right_idx, k] - tree.right_bound[k]) + tree.right_bound[k]
        return x_copy

def treecdf(tree: treenode.KeyTreeNode, pleft: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if not tree.left or not tree.right:
        return residual
    else:
        k = tree.split_axis
        x_new = np.ones_like(residual) * 2.
        left_idx = residual[:,k]<=tree.split_point
        right_idx = residual[:,k]>tree.split_point
        x_new[left_idx,:]=treecdf(tree.left, pleft[left_idx,:], residual[left_idx,:])
        x_new[right_idx,:]=treecdf(tree.right, pleft[right_idx,:], residual[right_idx,:])
        return treecdf_A(tree, pleft, x_new)

def log_density(tree:treenode.KeyTreeNode, pleft: np.ndarray, residual: np.ndarray) -> np.ndarray:
    ld = np.zeros((residual.shape[0],1))
    if not tree.left or not tree.right:
        return ld
    else:
        k = tree.split_axis
        key = int(tree.key)
        mu_left = np.log(tree.split_point - tree.left_bound[k]) - np.log(tree.right_bound[k] - tree.left_bound[k])
        mu_right = np.log(tree.right_bound[k] - tree.split_point) - np.log(tree.right_bound[k] - tree.left_bound[k])
        left_idx = residual[:,k]<=tree.split_point
        right_idx = residual[:,k]>tree.split_point   
        ld[left_idx,:] = log_density(tree.left, pleft[left_idx,:], residual[left_idx,:]) + np.log(pleft[left_idx,key]).reshape(-1,1) - mu_left
        ld[right_idx,:] = log_density(tree.right, pleft[right_idx,:],residual[right_idx,:]) + np.log(1 - pleft[right_idx,key]).reshape(-1,1) - mu_right
        return ld
    
def residualization_logdensity_helper(residual: np.ndarray, X: np.ndarray, tree: treenode.KeyTreeNode, c0: float, gamma: float):
    pleft = model2prob(tree, X)
    pleft = scale_dependent_shrinkage(tree, c0, gamma, pleft)
    res_new = treecdf(tree, pleft, residual)
    ld = log_density(tree, pleft, residual)
    return (res_new, ld)

def MC(X: np.ndarray, trees, c0: float, gamma: float, d, save_path = False, verbose = False):
    # X is 1*k
    # trees: List(treenode.KeyTreeNode)
    if save_path:
        u_path = []
    U = np.random.uniform(size = (d,))
    u_path.append(U)
    for __, tree in enumerate(trees[::-1]):
        if verbose:
            print("iteration: " + str(__), flush = True)
        treex = treenode.KeyTreeNode.from_tree_node(tree)
        treex.set_keys_bfs('')
        pleft = model2prob(treex, X)
        pleft = scale_dependent_shrinkage(treex, c0, gamma, pleft)
        treex.setp_bfs(pleft)
        U = treex.invCDF_recursion(U)
        u_path.append(U)
    if save_path:
        return u_path
    else:
        return U


def MC_reverse(U, X: np.ndarray, trees, c0: float, gamma: float, d):
    # X is 1*k
    # trees: List(treenode.KeyTreeNode)
    # U = np.random.uniform(size = (d,))
    for tree in trees[::-1]:
        treex = treenode.KeyTreeNode.from_tree_node(tree)
        treex.set_keys_bfs('')
        pleft = model2prob(treex, X)
        pleft = scale_dependent_shrinkage(treex, c0, gamma, pleft)
        treex.setp_bfs(pleft)
        U = treex.invCDF_recursion(U)
    return U
    
