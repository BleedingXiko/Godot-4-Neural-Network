class_name DeepQ

var observation_space: int
var action_spaces: int
var neural_network: NeuralNetworkAdvanced

# Hyper-parameters
var exploration_probability: float = 1.0
var exploration_decreasing_decay: float = 0.01
var min_exploration_probability: float = 0.05
var discounted_factor: float = 0.9
var learning_rate: float = 0.2
var decay_per_steps: int = 100

# Variables for tracking steps and learning
var steps_completed: int = 0
var is_learning: bool = true

# Variables to keep the previous state and action
var previous_state: Array = []
var previous_action: int

func _init(n_features: int, n_nodes: Array, n_action_spaces: int, hidden, output, _is_learning: bool = true) -> void:
	observation_space = n_features  # Number of features in the state
	action_spaces = n_action_spaces
	is_learning = _is_learning
	
	# Initialize the neural network with the number of features as input nodes
	neural_network = NeuralNetworkAdvanced.new(n_features, n_nodes, action_spaces, hidden, output)

func predict(current_states: Array, reward_of_previous_state: float) -> int:
	var current_q_values = neural_network.predict(current_states)
	
	if is_learning and previous_state.size() != 0:
		#var old_q_value = neural_network.predict(previous_state)[previous_action]
		var max_future_q = current_q_values.max()
		var target_q_value = reward_of_previous_state + discounted_factor * max_future_q
		var target_q_values = neural_network.predict(previous_state)
		target_q_values[previous_action] = target_q_value
		neural_network.train(previous_state, target_q_values)
	
	var action_to_take: int
	if randf() < exploration_probability:
		action_to_take = randi() % action_spaces
	else:
		action_to_take = current_q_values.find(current_q_values.max()) # needs the indexnot the value

	if is_learning:
		previous_state = current_states
		previous_action = action_to_take
	
	if steps_completed % decay_per_steps == 0:
		exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
	
	steps_completed += 1
	return action_to_take
