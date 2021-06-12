# This is the implementation of the following paper
Dudek G.: A Constructive Approach to Data-Driven Randomized Learning for Feedforward Neural Networks. Applied Soft Computing (submitted)

Abstract: 
Feedforward neural networks with random hidden nodes have a problem generating random parameters as these are difficult to set optimally to obtain a good projection space. Typically, random weights and biases are both drawn from the same interval, which is misguided as they have different functions. Recently, some sophisticated methods of random parameters generation have been developed, such as the data-driven method, where the sigmoids are placed in randomly selected regions of the input space and then their slopes are adjusted to the local fluctuations of the target function. In this work, we propose a new constructive data-driven method that constructs iteratively the network architecture. This method successively generates new candidate hidden nodes and accepts them if the training error decreases significantly. The threshold of acceptance is adapted during training, accepting at the beginning of the training process only those nodes which lead to the largest error reduction. In the next stages, the threshold is successively reduced to accept those nodes which model the target function details more accurately. This leads to a more compact network architecture, as it includes only "significant" nodes. The redundant, random nodes, which are usually generated by existing randomized learning methods, are not accepted by the proposed method.  
We empirically compared our approach against several alternative methods, including its predecessor, competitive randomized learning solutions, a gradient-based network and a generalized additive model. We found that our proposed approach has superior performance and outperforms its competitors in fitting accuracy.

Research highlights
- A new way of generating hidden node parameters in FNN randomized learning is proposed
- Constructive data-driven approach builds compact NNs with "relevant" nodes
- Hidden nodes generated by CD-DM reflects local complexity of the target function

Keywords: 
data-driven randomized learning, feedforward neural networks, neural networks with random hidden nodes, randomized learning algorithms
