class_name QLearningDev

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int
var action_spaces: int

# The neural network for Q-learning
var q_network: NeuralNetworkAdvanced

# Hyper-parameters
var exploration_probability: float = 1.0
var exploration_decreasing_decay: float = 0.01
var min_exploration_probability: float = 0.01
var discounted_factor: float = 0.9
var learning_rate: float = 0.2
var decay_per_steps: int = 100
var steps_completed: int = 0

# States
var previous_state: int = -100
var current_state: int
var previous_action: int

var done: bool = false
var is_learning: bool = true
var print_debug_info: bool = false

# List to store experiences for training the neural network
var experience_buffer: Array = []

func _init(n_observations: int, n_action_spaces: int, _is_learning: bool = true) -> void:
    observation_space = n_observations
    action_spaces = n_action_spaces
    is_learning = _is_learning
    
    q_network = NeuralNetworkAdvanced.new()
    q_network.add_layer(nodes: observation_space, activation: ACTIVATIONS.RELU)  # Input layer
    q_network.add_layer(nodes: 64, activation: ACTIVATIONS.RELU)  # Hidden layer
    q_network.add_layer(nodes: action_spaces)  # Output layer

func add_experience(state: int, action: int, reward: float, next_state: int) -> void:
    experience_buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state})

func train_neural_network() -> void:
    for experience in experience_buffer:
        var state = experience["state"]
        var action = experience["action"]
        var reward = experience["reward"]
        var next_state = experience["next_state"]

        var input_array = Matrix.new(state, 1)  # Adjust based on your Matrix class
        var next_input_array = Matrix.new(next_state, 1)  # Adjust based on your Matrix class

        var q_values = q_network.predict(input_array)
        var target_q_value = reward + discounted_factor * q_network.predict(next_input_array).max_from_row(0)
        q_values[action] = (1 - learning_rate) * q_values[action] + learning_rate * target_q_value

        q_network.train(input_array, q_values)

    experience_buffer.clear()

func predict(current_state: int, reward_of_previous_state: float) -> int:
    if is_learning:
        if previous_state != -100:
            add_experience(previous_state, previous_action, reward_of_previous_state, current_state)
            train_neural_network()

    var action_to_take: int
    
    if randf() < exploration_probability and is_learning:
        action_to_take = randi_range(0, action_spaces - 1)
    else:
        var input_array_exploit = Matrix.new(current_state, 1)  # Adjust based on your Matrix class
        var q_values = q_network.predict(input_array_exploit)
        action_to_take = q_values.index_of_max_from_row(0)  # Assuming q_values is a column vector
    
    if is_learning:
        previous_state = current_state
        previous_action = action_to_take
    
        if steps_completed != 0 and steps_completed % decay_per_steps == 0:
            exploration_probability = maxf(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
    
    if print_debug_info and steps_completed % decay_per_steps == 0:
        print("Total steps completed:", steps_completed)
        print("Current exploration probability:", exploration_probability)
        print("Q-Values:", q_values.data)  # Assuming q_values is your Matrix class
        print("-----------------------------------------------------------------------------------------")
    
    steps_completed += 1
    return action_to_take