from __future__ import division
from __future__ import print_function
import sys
if sys.version.startswith('3'):
    xrange = range
import numpy as np
rand = np.random.rand
from scipy.spatial.distance import pdist, squareform


class SSO_NN(object):
    """
    A two-layer fully-connected neural network trained using Social Spider Algorithm (SSA) [1]. 
    The net has an input dimension of N, a hidden layer dimension of H, and 
    performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first and the 
    second fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - ReLU - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    
    [1] James J.Q. Yu and Victor O.K. Li, "A Social Spider Algorithm for Global Optimization," Appl. Soft Comput., vol. 30, pp. 614–627, 2015.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 pop_size,
                 std = 1e-2,
                 reg = 0.05,
                 r_a = 1,
                 p_c = 0.7,
                 p_m = 0.1,
                ):
        """
        Initialize the model. Weights and biases are initialized to small random values. Weights and biases are stored in a variable
        self.position, which is an array of (pop_size) vectors, each vector represents the position of individual spider, which 
        represents a possible solution for the values of the weights and biases of the network.
        Each position vector contains two weights and two biases encoded in the following order:

        [:(D * H)] => W1 First layer weights; has shape (D, H)
        [(D * H):((D * H) + (H * C))] => W2 Second layer weights; has shape (H, C)
        [-(H + C):] => b1 + b2 First and Second layer biases; has shape (H,) and (C,)
        Where:
        - D: input_size
        - H: hidden_size
        - C: output_size

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        - pop_size: Number of soiders (Population size)
        - reg: The strength of L2 regularization
        - std: User-controled parameter to initialized small values
        - r_a: User-controled parameter that controles the attenuation rate of vibration intensity over distance
        - p_c: User-controled parameter that describes the probability of changing mask
        - p_m: Each bit in the mask has a probability of p_m to assigned with a one, and (1-p_m) with zero 
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dim = (input_size * hidden_size) + (hidden_size * output_size) + hidden_size + output_size
        self.pop_size = pop_size
        self.std = std
        self.reg = reg
        self.r_a = r_a
        self.p_c = p_c
        self.p_m = p_m
        #Set global best loss to infinity
        self.g_best = np.Inf
        self.g_best_hist = []
        #Set global best position to zeros
        self.g_best_pos = np.zeros(self.dim)
        # Create the position of population of spiders pop in a search space of dimension (D*H + H*C + H + C).
        self.position = self.std*np.random.randn(self.pop_size, self.dim)
            
    
    def loss(self, X, y):
        """
        Compute the loss with regularization for a two layer fully connected neural
        network in each spider position.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C.

        Returns:
        - loss_list: A 1-D array (pop_size) of losses for each sipder in the 
        population over all training samples.
        """
        loss_list = []
        for pos in self.position:
            w1, w2, b1, b2 = self.decode_position(pos)
            
            # Compute the forward pass
            hidden_layer = np.maximum(0, np.dot(X, w1) + b1) # note, ReLU activation
            scores = np.maximum(0, np.dot(hidden_layer, w2) + b2)
            
            #######################################
            
            # Compute the loss
            reg_loss = self.reg*(np.sum(np.square(w1)) + np.sum(np.square(w2)))
            num_examples = X.shape[0]
            # get unnormalized probabilities
            exp_scores = np.exp(scores)
            # normalize them for each example
            probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1E-100)
            # compute the loss: average cross-entropy loss and regularization
            correct_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(correct_logprobs)/num_examples
            loss = data_loss + reg_loss
            loss_list.append(loss)

        return np.asarray(loss_list)
    
    def decode_position(self, position):
        """
        Decode the weights and biases of the network from one spider position.
        *This function is called inside of a loop of all spiders

        Inputs:
        - position: The position of one spider from list of the whole population, 
        which represents a possible solution for the network.

        Returns:
        - W1: First layer weights; has shape (D, H)
        - W2: Second layer weights; has shape (H, C)
        - b1: First layer biases; has shape (H,)
        - b2: Second layer biases; has shape (C,)
        """
        
        w1 = position[:(self.input_size*self.hidden_size)]
        w2 = position[(self.input_size*self.hidden_size):((self.input_size*self.hidden_size) + (self.hidden_size*self.output_size))]
        b = position[-(self.hidden_size + self.output_size):]
        b1 = b[:self.hidden_size]
        b2 = b[self.hidden_size:]
        return w1.reshape(self.input_size, self.hidden_size) ,w2.reshape(self.hidden_size, self.output_size), b1, b2
            
    def g_best_pos_acc(self, X, y):
        w1, w2, b1, b2 = self.decode_position(self.g_best_pos)
        hidden_layer = np.maximum(0, np.dot(X, w1) + b1) # note, ReLU activation
        scores = np.maximum(0, np.dot(hidden_layer, w2) + b2)
        predicted_class = np.argmax(scores, axis=1)
        return np.mean(predicted_class == y)

    def train(self, X, y, max_iteration, show_info = False):
        """
        Train the neural network using Social Spider Algorithm.
        
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C.
        - max_iteration: Maximum number of iterations for the algorithm to run
        
        Returns:
        A dictionary contains the following keys:
        - global_best_loss: Optimum (Minimum) loss of the network achieved by the colony
        - global_best_acc: Optimum (Maximum) accuracy of the network achieved by the colony
        - W1: Optimum values of the first layer weights; has shape (D, H)
        - W2: Optimum values of the second layer weights; has shape (H, C)
        - b1: Optimum values of the first layer biases; has shape (H,)
        - b2: Optimum values of the second layer biases; has shape (C,)
        """
        # Initialize the target position for each spider.
        target_position = self.position.copy()
        # Initialize the target vibration intensity for each spider.
        target_intensity = np.zeros(self.pop_size)
        
        mask = np.zeros((self.pop_size, self.dim))
        movement = np.zeros((self.pop_size, self.dim))
        inactive = np.zeros(self.pop_size)

        
        if show_info:
            import datetime, time
            print(" " * 15 + "SSA starts at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print("=" * 75)
            print(" iter optimum_loss optimum_acc  pop_min  base_dist  mean_dist time_elapsed")
            print("=" * 75)
            self.start_time = time.clock()
        
        iteration = 0
        while (iteration < max_iteration):
            iteration += 1
            # Evaluate the fitness value of each spider s in pop
            spider_fitness = self.loss(X, y)
            # Calculate the standard deviation of all spider positions along each dimension
            base_distance = np.mean(np.std(self.position, 0))
            
            #distance=>shape (pop×pop), calculate the 1-norm (Manhattan) distance between all spiders
            distance = squareform(pdist(self.position, 'cityblock'))
            
            #Update global best loss and position
            if np.min(spider_fitness) < self.g_best:
                self.g_best = np.min(spider_fitness)
                self.g_best_pos = self.position[np.argmin(spider_fitness)].copy()
            self.g_best_hist.append(self.g_best)
            # If we are in the first itration, do the following:
            if show_info and (iteration == 1 or iteration == 10
                    or (iteration < 1001 and iteration % 100 == 0) 
                    or (iteration < 10001 and iteration % 1000 == 0)
                    or (iteration < 100000 and iteration % 10000 == 0)):
                elapsed_time = time.clock() - self.start_time
                print(repr(iteration).rjust(5), "%.4e" % self.g_best,"%.4e" % self.g_best_pos_acc(X, y),
                      "%.4e" % np.min(spider_fitness),
                      "%.4e" % base_distance, "%.4e" % np.mean(distance), 
                      "%02d:%02d:%02d.%03d" % (elapsed_time // 3600, elapsed_time // 60 % 60, 
                                               elapsed_time % 60, (elapsed_time % 1) * 1000))
            

            # Calculate the vibration intensity at the source position
            intensity_source = np.log(1. / (spider_fitness + 1E-100) + 1)
            # Calculate the vibration attenuation over distance
            intensity_attenuation = np.exp(-distance / (base_distance * self.r_a))
            # Calculate the vibration intensity received by each spider from all spiders in the colony
            intensity_receive = np.tile(intensity_source, self.pop_size).reshape(self.pop_size, self.pop_size) * intensity_attenuation
            
            # Select the strongest vibration
            max_index = np.argmax(intensity_receive, axis = 1)
            # Decide if to change the target position, and updating it if yes
            keep_target = intensity_receive[np.arange(self.pop_size),max_index] <= target_intensity
            keep_target_matrix = np.repeat(keep_target, self.dim).reshape(self.pop_size, self.dim)
            inactive = inactive * keep_target + keep_target
            target_intensity = target_intensity * keep_target + intensity_receive[np.arange(self.pop_size),max_index] * (1 - keep_target)
            target_position = target_position * keep_target_matrix + self.position[max_index] * (1 - keep_target_matrix)

            # Select a random position from population positions and repeat it pop_size times
            rand_position = self.position[np.floor(rand(self.pop_size * self.dim) * self.pop_size).astype(int), \
                np.tile(np.arange(self.dim), self.pop_size)].reshape(self.pop_size, self.dim)

            # Update the dimension mask m_s.
            new_mask = np.ceil(rand(self.pop_size, self.dim) + rand() * self.p_m - 1)
            keep_mask = rand(self.pop_size) < self.p_c**inactive
            inactive = inactive * keep_mask
            keep_mask_matrix = np.repeat(keep_mask, self.dim).reshape(self.pop_size, self.dim)
            mask = keep_mask_matrix * mask + (1 - keep_mask_matrix) * new_mask
                            
            # Generate a new following position
            follow_position = mask * rand_position + (1 - mask) * target_position
            # performs a random walk to the following position.
            movement = np.repeat(rand(self.pop_size), self.dim).reshape(self.pop_size, self.dim) * movement + \
                (follow_position - self.position) * rand(self.pop_size, self.dim)
            self.position = self.position + movement
            
        if show_info:
            elapsed_time = time.clock() - self.start_time
            print("=" * 75)
            print(repr(iteration).rjust(5), "%.4e" % self.g_best,"%.4e" % self.g_best_pos_acc(X, y), "%.4e" % np.min(spider_fitness),
                  "%.4e" % base_distance, "%.4e" % np.mean(distance), 
                  "%02d:%02d:%02d.%03d" % (elapsed_time // 3600, elapsed_time // 60 % 60, 
                                           elapsed_time % 60, (elapsed_time % 1) * 1000))
            print("=" * 75)
        w1, w2, b1, b2 = self.decode_position(self.g_best_pos)
        return {'global_best_loss': self.g_best,
                'global_best_acc': self.g_best_pos_acc(X, y),
                'W1': w1,
                'W2': w2,
                'b1': b1,
                'b2': b2}

