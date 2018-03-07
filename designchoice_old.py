import math, numpy as np

class DesignChoice:
    # Input parameters (sent during class initialisation) - network depth, input length d0, number of epochs
    # Output - hidden layer sizes, batch size?
    # Select the best (bound?) network architecture based on the expectation of the gradients

    def __init__(self,
                 phis,
                 vgg_transformed=True,
                 new_input_length=3,
                 depth = 5,
                 data_type = 'MNIST',
                 max_epochs = 81,
                 logbase = 10):
        #TODO : Activation function type

        self.max_epochs = max_epochs
        self.data_type = data_type
        self.network_depth = depth
        self.psi = -1
        self.dl = -1
        self.logbase = logbase

        if data_type == 'MNIST' or data_type == 'mrbi':
            self.d0 = 28*28
            self.dl = 10

        if data_type == 'cifar10' or data_type == 'svhn':
            if vgg_transformed:
                if new_input_length == 1:
                    self.d0 = 16384
                if new_input_length == 2:
                    self.d0 = 8192
                if new_input_length == 3:
                    self.d0 = 4096
                if new_input_length == 4:
                    self.d0 = 2048
                if new_input_length == 5:
                    self.d0 = 1024
            else:
                self.d0 = 32*32*3
                self.dl = 10

        # Setup constants, phis
        self.phis = phis

        self.hidden_layers = []


    # Initialise the matrices and vectors based on 3 conditions - o/p length and psi given/fixed, either one given, neither given
    # Matrix - X - the weight matrix associated with the coefficients in the linear equations
    # Vector - h - The hidden layer sizes of each layer
    # Vector - c - The constants from phis

    # TODO : This works only for depths >= 5. For lesser, need to solve linear equations sequentially

    def get_hidden_layers(self, phi_index = 0, psi_index = 0, flag = 1): # flag = 1, dls are transferrable; flag = 2, dls aren't transferrable

         if self.dl > -1:
            if self.psi == -1:
                # Just solve for the last equation
                self.psi = math.sqrt(self.dl*self.phis[phi_index][self.network_depth-1])
                print(self.psi)

         else: # Configured through constructor?
            if self.psi == -1:
                self.psi = 0.5 # default
            self.dl = self.psi**2/self.phis[phi_index][self.network_depth-1]

         self.psi = 0.5 #TODO: Fix this hard coding

         if self.network_depth == 3:  # No need to make this into a matrix
            dl_minus1 = int(math.ceil(float(self.psi) / self.phis[phi_index][1] / self.dl))
            if dl_minus1 == 0:
                dl_minus1 = 1
            if dl_minus1 >= 10000:
                dl_minus1 = 1000

            dl_plus1 = int(math.ceil(1.0 / self.phis[phi_index][0] / self.psi / self.d0 / dl_minus1))
            if dl_plus1 == 0:
                dl_plus1 = 1
            if dl_plus1 >= 10000:
                dl_plus1 = 1000


            h = []
            h.append(dl_plus1)
            h.append(dl_minus1)

            return h

         if self.network_depth >= 5:
            X = np.zeros((self.network_depth - 2, self.network_depth - 2))
            c = np.zeros((self.network_depth - 2,1))
            for i in range(self.network_depth - 3):
                c[i] = math.log(1.0/self.phis[phi_index][i], self.logbase) - math.log(self.psi, self.logbase)
                if (i == 0):
                     c[i] = c[i] - math.log(self.d0, self.logbase)

                     # i = 0 is a special case where only d1 and d2 are unknowns
                     X[0][0] = 1
                     X[0][1] = 1

                else:
                     # X follows pattern that i, i-1 and i-2 are 1's
                     X[i][i-1] = 1
                     X[i][i] = 1
                     X[i][i+1] = 1

            # L-1 can be found
            print self.psi, self.phis[phi_index][self.network_depth-2], self.dl
            dl_minus1 = int(float(self.psi)/self.phis[phi_index][self.network_depth-2]/self.dl)
            if dl_minus1 == 0:
                dl_minus1 = 1

            # L-2 layer is different - has only L-2 and L-3
            X[self.network_depth-3][self.network_depth-3-1] = 1
            X[self.network_depth-3][self.network_depth-3] = 1

            # L-2 layer's constant is appended with log(dl-1)
            c[self.network_depth-3] = math.log(1.0/self.phis[phi_index][self.network_depth-3], self.logbase) - math.log(self.psi, self.logbase) - math.log(dl_minus1, self.logbase)

            h_minus2 = np.matmul(np.linalg.inv(X), c)

            for i in range(len(h_minus2)):
                h_minus2[i] = int(10**h_minus2[i])
                if h_minus2[i] == 0:
                    h_minus2[i] = 1

            h = np.resize(h_minus2, (self.network_depth-1, 1))
            h[self.network_depth-2] = int(dl_minus1)

            return h

    def get_all_hidden_layers(self):
        for i in range(len(self.phis)):
            self.hidden_layers.append(self.get_hidden_layers(phi_index=i))
        return self.hidden_layers















