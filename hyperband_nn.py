# Implementation of hyperband as in the paper : "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
# Given a hyper parameter space and resource constraints, learn the best hyper parameter set possible
# Here, the resource constraint is the number of epochs/iterations

import math, numpy


class HyperbandNN:
    # Parameters - (Things in () are equivalent terminology in the paper)
    # get_from_hyperparam_space : Function that samples from hyperparam space (get_random_hyperparameter_configuration)
    # run_eval_hyperparam : Function that runs and evaluates chosen hyperparams (run_then_return_val_loss)
    # max_epochs : Maximum epochs/iterations available per configuration ('R')
    # eta : downsampling rate

    def __init__(self,
                 get_from_hyperparam_space,
                 run_eval_hyperparam,
                 max_epochs=10,
                 eta=4,
                 keepProb=0.5):

        #TODO: Specify different resources

        # Functions
        self.get_from_hyperparam_space = get_from_hyperparam_space
        self.run_eval_hyperparam = run_eval_hyperparam

        # Constants
        self.max_epochs = max_epochs
        self.eta = eta  # Default is 3
        self.s_max = int(math.log(max_epochs, self.eta))  # Same as s_max in the paper
        self.B = (self.s_max + 1) * self.max_epochs  # Same as 'B' in the paper
        self.keep_prob = keepProb

        self.outcomes = []  # List of dictionaries to keep track of each result after every run of s

    def run(self, debug=False):
    #TODO: Multiple resources and then ensemble

        for s in reversed(range(self.s_max + 1)):

            # Initialize number of configurations to consider in this run of s and max num of iterations to run configs for
            n = int(math.ceil((self.B / self.max_epochs / (s + 1) * self.eta ** s)))
            r = self.max_epochs * self.eta ** (-s)

            # Successive halving with (n, r)

            # First get n configurations from hyperparam space
            T = [self.get_from_hyperparam_space() for i in range(n)]
            print n, self.B, self.max_epochs, self.s_max, s+1, self.eta, s, len(T)
            # Successively half the configs

            result = {}
            for i in range(s+1):

                # Run each n_i configs for r_i iterations
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** (i)

                # Run all hyperparam configs for
                loss = [self.run_eval_hyperparam(t, int(r_i), self.keep_prob) for t in T]

                if debug:
                    print('Loss for r_i = %f - n_i = %f ' % (r_i, n_i)), 
                    print 'and hyper parameters - ',
                    all_t_lr = [T_i.learning_rate for T_i in T]
                    all_t_hls = [T_i.hidden_layer_sizes for T_i in T]
                    print all_t_lr
                    print all_t_hls, 'is - '
                    print (loss)

                T = [T[i] for i in numpy.argsort(loss)[0:int(math.ceil(float(n_i) / self.eta))]]

                # Store the best result
                result['s'] = s
                result['n_i'] = n_i
                result['r_i'] = r_i
                result['error'] = min(loss)
                result['hyperparams_learning_rate'] = T[0].learning_rate
                result['hyperparams_hidden_layer_sizes'] = T[0].hidden_layer_sizes

            self.outcomes.append(result)
