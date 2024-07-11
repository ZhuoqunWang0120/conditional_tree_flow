import fast_func
import treenode
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import os
import pickle

class TreeEnsemble:
    def __init__(self):
        self.trees = []
        self.residual_train = []
        self.residual_validation = []
        self.validation_log_density_increment = None
        self.stopped_at = []
    
    def fit(self, X, Y, niter, model_name, model_args, quiet, c0, gamma, X_validation = None, y_validation = None, validation_fraction = 0.1, validation_rolling_window = 20, validation_inc_threshold = 0., mode = None, cache_dir = None, **kwargs):
        assert X.shape[0] == Y.shape[0]
        if cache_dir:
            os.system('mkdir -p ' + cache_dir)
        if not mode:
            mode = 'absolute'
        if (X_validation is None) or (y_validation is None):
            X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=validation_fraction, random_state=42)
        else:
            X_train, y_train = X, Y
        n_validation = X_validation.shape[0]
        if n_validation > 0:
            validation_log_density_increment = np.empty(shape=(n_validation, niter))
        # fit estimators
        dim = y_train.shape[1]
        residual_train = []
        residual_validation = []
        res_train = y_train.copy()
        res_validation = y_validation.copy()
        # trees = ()
        for it in range(niter):
            if not quiet:
                print(model_name, flush=True)
                print(it, flush=True)
            # fit G_k, T_k
            tree = treenode.TreeNode(left_bound=np.zeros(shape = (dim,)), right_bound=np.ones(shape = (dim,)), dim = dim)
            kwargs.update({'residual': res_train, 'X': X_train, 'model_name': model_name, 'model_args': model_args})
            # getattr(tree, learner)(**kwargs) # tree.[learner](**kwargs)
            tree.fit_greedy_regularization_dens_multiprocessing(**kwargs)
            self.trees.append(copy.deepcopy(tree))
            # trees.append(copy.deepcopy(tree))
            # update residuals
            # training
            tree = treenode.KeyTreeNode.from_tree_node(tree)
            tree.set_keys_bfs('')
            res_train, log_density_train = fast_func.residualization_logdensity_helper(residual=res_train, X=X_train, tree=tree, c0=c0, gamma=gamma)
            residual_train.append(res_train)
            # validation density and early stopping
            if n_validation > 0:
                res_validation, log_density_validation = fast_func.residualization_logdensity_helper(res_validation, X_validation, tree, c0, gamma)
                residual_validation.append(res_validation)
                validation_log_density_increment[:,it] = log_density_validation.reshape(-1)
                if it > validation_rolling_window:
                    if ((mode == 'absolute') and (np.mean(validation_log_density_increment[:,(it - validation_rolling_window + 1):(it + 1)]) < validation_inc_threshold)) or ((mode == 'relative') and (np.sum(validation_log_density_increment[:,(it - validation_rolling_window + 1):(it + 1)]) / np.abs(np.sum(validation_log_density_increment[:,:(it - validation_rolling_window + 1)])) < validation_inc_threshold)):
                        break    
            if cache_dir:
                with open(cache_dir + '/' + model_name + str(it) + '.pkl', 'wb') as cache_file:
                    pickle.dump({'tree': tree, 'res_train': res_train, 'res_validation': res_validation, 'log_density_train': log_density_train, 'log_density_validation': log_density_validation}, cache_file)
                    cache_file.close()
            
        self.residual_train = residual_train[:-1]
        self.residual_validation = residual_validation[:-1]
        self.trees = self.trees[:-1]
        self.validation_log_density_increment = validation_log_density_increment[:,:it]
        self.stopped_at = [it]
        self.data_split = (X_train, X_validation, y_train, y_validation)

    def append(self, ensemble_inst):
        self.residual_train.extend(ensemble_inst.residual_train)
        self.residual_validation.extend(ensemble_inst.residual_validation)
        self.trees.extend(ensemble_inst.trees)
        self.stopped_at.extend(ensemble_inst.stopped_at)
        if self.validation_log_density_increment is None:
            self.validation_log_density_increment = ensemble_inst.validation_log_density_increment
        else:
            self.validation_log_density_increment = np.concatenate([self.validation_log_density_increment, ensemble_inst.validation_log_density_increment], axis = 1)
            
                
