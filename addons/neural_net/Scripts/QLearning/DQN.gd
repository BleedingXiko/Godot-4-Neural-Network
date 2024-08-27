class_name DQN

var neural_network: Object  # Can be either NeuralNetworkAdvanced or NeuralNetworkInNetwork

# Hyper-parameters
var use_target_network: bool = true
var target_neural_network: Object  # Can be either NeuralNetworkAdvanced or NeuralNetworkInNetwork
var update_target_every_steps: int = 1000

var exploration_probability: float = 1.0
var exploration_decreasing_decay: float = 0.01
var min_exploration_probability: float = 0.05
var discounted_factor: float = 0.9
var learning_rate: float = 0.2
var use_l2_regularization: bool = true
var l2_regularization_strength: float = 0.001
var decay_per_steps: int = 100
var print_debug_info: bool = false

# Exploration strategy: "epsilon_greedy" or "softmax"
var exploration_strategy: String = "epsilon_greedy"

# Replay memory
var memory_capacity: int = 300
var replay_memory: Array = []
var batch_size: int = 30

# Variables for tracking steps and learning
var steps_completed: int = 0
var is_learning: bool = true
var use_replay: bool = false

# Variables to keep the previous state and action
var previous_state: Array = []
var previous_action: int

func _init(config: Dictionary) -> void:
	# Configuring hyper-parameters from the config dictionary
	set_config(config)
	
	# Initialize the neural network based on the configuration
	if config.has("use_nin") and config["use_nin"]:
		neural_network = NeuralNetworkInNetwork.new(config)
	else:
		neural_network = NeuralNetworkAdvanced.new(config)
	
	# Initialize the target network if required
	if use_target_network:
		if config.has("use_nin") and config["use_nin"]:
			target_neural_network = NeuralNetworkInNetwork.new(config)
		else:
			target_neural_network = neural_network.copy()

func set_config(config: Dictionary) -> void:
	exploration_probability = config.get("exploration_probability", exploration_probability)
	exploration_decreasing_decay = config.get("exploration_decreasing_decay", exploration_decreasing_decay)
	min_exploration_probability = config.get("min_exploration_probability", min_exploration_probability)
	discounted_factor = config.get("discounted_factor", discounted_factor)
	decay_per_steps = config.get("decay_per_steps", decay_per_steps)
	use_replay = config.get("use_replay", use_replay)
	is_learning = config.get("is_learning", is_learning)
	learning_rate = config.get("learning_rate", learning_rate)
	use_target_network = config.get("use_target_network", use_target_network)
	update_target_every_steps = config.get("update_target_every_steps", update_target_every_steps)
	memory_capacity = config.get("memory_capacity", memory_capacity)
	batch_size = config.get("batch_size", batch_size)
	use_l2_regularization = config.get("use_l2_regularization", use_l2_regularization)
	l2_regularization_strength = config.get("l2_regularization_strength", l2_regularization_strength)
	decay_per_steps = config.get("decay_per_steps", decay_per_steps)
	print_debug_info = config.get("print_debug_info", print_debug_info)
	exploration_strategy = config.get("exploration_strategy", exploration_strategy)

func add_dense(nodes: int, function: Dictionary = neural_network.ACTIVATIONS.SIGMOID, input_shape: int = 0):
	# Add a dense (regular) layer to the neural network
	if neural_network is NeuralNetworkAdvanced:
		neural_network.add_layer(nodes, function)
	else:
		print("Cannot add a dense layer to a Network-in-Network setup directly. Use add_master_network_layer.")

func add_nin(nodes: int, sub_hidden_layers: Array = [1], input_shape: int = 0):
	# Add a NIN layer to the neural network
	if neural_network is NeuralNetworkInNetwork:
		var previous_nodes: int
		
		if neural_network.sub_networks.size() > 0:
			# Get the number of sub-networks in the last NIN layer (this acts as the input size for the next layer)
			previous_nodes = neural_network.sub_networks[-1].size()
		else:
			# If there are no sub-networks, use the provided input_shape as the input size for the first layer
			previous_nodes = input_shape

		# Add the NIN layer with the correct input size
		neural_network.add_nin_layer(nodes, previous_nodes, sub_hidden_layers)
	else:
		print("Cannot add a NIN layer to a dense network setup.")

func add_master_network_layer(output_size: int, master_hidden_layers: Array = [], activation: Dictionary = neural_network.ACTIVATIONS.SIGMOID):
	# Add the master network for NIN
	if neural_network is NeuralNetworkInNetwork:
		neural_network.add_master_network_layer(output_size, master_hidden_layers, activation)
	else:
		print("Master network layer is only applicable to NIN setups.")

func add_to_memory(state, action, reward, next_state, done):
	if replay_memory.size() >= memory_capacity:
		replay_memory.pop_front()
	replay_memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})

func sample_memory():
	var batch = []
	for i in range(min(batch_size, replay_memory.size())):
		batch.append(replay_memory.pick_random())
	return batch

func update_target_network():
	if use_target_network:
		print("updated Target Network")
		target_neural_network = neural_network.copy()

func train_batch(batch):
	for experience in batch:
		var max_future_q: float
		if use_target_network:
			max_future_q = target_neural_network.predict(experience["next_state"]).max()
		else:
			max_future_q = neural_network.predict(experience["next_state"]).max()
		var target_q_value = experience["reward"] + discounted_factor * max_future_q if not experience["done"] else experience["reward"]
		var target_q_values = neural_network.predict(experience["state"])
		target_q_values[experience["action"]] = target_q_value
		neural_network.train(experience["state"], target_q_values)

func softmax(q_values: Array) -> Array:
	var exp_values = []
	var sum_exp = 0.0
	for q in q_values:
		var exp_q = exp(q)
		exp_values.append(exp_q)
		sum_exp += exp_q
	var probabilities = []
	for exp_q in exp_values:
		probabilities.append(exp_q / sum_exp)
	return probabilities

func choose_action_softmax(q_values: Array) -> int:
	var probabilities = softmax(q_values)
	var cumulative_prob = 0.0
	for i in range(probabilities.size()):
		cumulative_prob += probabilities[i]
		if randf() < cumulative_prob:
			return i
	return probabilities.size() - 1

func train(current_states: Array, reward_of_previous_state: float, done: bool = false) -> int:
	var current_q_values = neural_network.predict(current_states)
	var current_action = choose_action(current_states)
	
	# Handle the learning and updating process
	if previous_state.size() != 0:
		if use_replay:
			add_to_memory(previous_state, previous_action, reward_of_previous_state, current_states, done)
			if replay_memory.size() >= batch_size:
				var batch = sample_memory()
				train_batch(batch)
		else:
			var max_future_q: int
			if use_target_network:
				max_future_q = target_neural_network.predict(current_states).max()
			else:
				max_future_q = current_q_values.max()
			var target_q_value = reward_of_previous_state + discounted_factor * max_future_q
			var target_q_values = neural_network.predict(previous_state)
			target_q_values[previous_action] = target_q_value
			neural_network.train(previous_state, target_q_values)

	# Update previous state and action for the next step
	previous_state = current_states
	previous_action = current_action
	
	# Handle target network updates
	if steps_completed % update_target_every_steps == 0:
		update_target_network()

	if steps_completed % decay_per_steps == 0:
		if print_debug_info:
			print("Total steps completed:", steps_completed)
			print("Current exploration probability:", exploration_probability)
			print("Q-Net data:", neural_network.debug())
			print("-----------------------------------------------------------------------------------------")
	
	steps_completed += 1
	return previous_action

func choose_action(current_states: Array) -> int:
	var current_q_values = neural_network.predict(current_states)
	var action_to_take: int

	if exploration_strategy == "epsilon_greedy":
		if randf() < exploration_probability:
			action_to_take = randi() % current_q_values.size()
		else:
			action_to_take = current_q_values.find(current_q_values.max())
		exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
		
	elif exploration_strategy == "softmax":
		action_to_take = choose_action_softmax(current_q_values)
	else:
		action_to_take = current_q_values.find(current_q_values.max())

	return action_to_take

func save(path):
	neural_network.save(path)

func load(path, config: Dictionary = {"exploration_strategy": "softmax"}):
	neural_network.load(path)
	set_config(config)
	update_target_network()
