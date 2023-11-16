AI Algorithms for Godot 4
=========================

This project offers a suite of AI algorithms in GDScript for Godot 4, designed for various AI applications in game development.

Neural Net Genetic Algorithm
----------------------------

### Rules for Usage

When utilizing the `Neural Net` genetic algorithm, adhere to the following rules:

1.  Variable Naming: The variable for the Neural Network must be named `nn`.

    `var nn: NeuralNetwork`

    This is essential as the `Neural Net` algorithm functions only when the network is named `nn`.

2.  Variable Declaration: Do not assign any value to the `nn` variable upon declaration.


    `var nn: NeuralNetwork`

    The algorithm depends on `nn` being declared but not assigned.

3.  Handling AI/Player Removal: Use the `queue_free()` method for removing AI or player nodes.


    `Object.queue_free()`

    The `Neural Net` relies on the signal emitted when a node exits the tree to receive its fitness and neural network data.

NeuralNetwork Class
-------------------

### Overview

A basic neural network capable of customization, training, and prediction.

### Usage

#### Initialization

`var nn = NeuralNetwork.new(input_nodes, hidden_nodes, output_nodes)`

#### Prediction

`var output = nn.predict(input_array)`

NeuralNetworkAdvanced Class
---------------------------

### Overview

An advanced version of the Neural Network with more complex features.

### Usage

#### Initialization

`var nn_advanced = NeuralNetworkAdvanced.new({
    "learning_rate": 0.1,
    "use_l2_regularization": true,
    "l2_regularization_strength": 0.001
})
nn_advanced.add_layer(nodes, activation_function)`

#### Prediction

`var output = nn_advanced.predict(input_array)`

QTable Class
------------

### Overview

Implements the Q-Learning algorithm for reinforcement learning.

### Usage

#### Initialization

Include configuration parameters like learning rate, exploration probability, and decay settings.

`var qtable = QTable.new(n_observations, n_action_spaces, {
    "learning_rate": 0.2,
    "exploration_probability": 1.0,
    "exploration_decreasing_decay": 0.01,
    "min_exploration_probability": 0.05,
    "discounted_factor": 0.9,
    "decay_per_steps": 100,
    "max_state_value": 2,
    "is_learning": true
})`

#### Predicting Action

`var action = qtable.predict(current_states, reward_of_previous_state)`

QNetwork Class
--------------

### Overview

Combines neural networks with Q-Learning for complex decision-making.

### Usage

#### Initialization

Include configuration parameters like exploration settings, learning rate, and network architecture.

`var qnetwork = QNetwork.new({
    "exploration_probability": 1.0,
    "exploration_decreasing_decay": 0.01,
    "min_exploration_probability": 0.05,
    "discounted_factor": 0.9,
    "learning_rate": 0.2,
    "use_target_network": true,
    "update_target_every_steps": 1000,
    "memory_capacity": 300,
    "batch_size": 30,
    "use_l2_regularization": true,
    "l2_regularization_strength": 0.001,
    "decay_per_steps": 100,
    "is_learning": true
})
qnetwork.add_layer(nodes, activation_function)`

#### Predicting Action

`var action = qnetwork.predict(current_states, reward_of_previous_state)`

General Guidelines
------------------

These classes can be used individually or in combination, depending on the AI requirements of your Godot project. Initialize each class with a configuration dictionary tailored to your specific needs. The prediction functionalities enable these models to make decisions or analyze situations based on given inputs.

