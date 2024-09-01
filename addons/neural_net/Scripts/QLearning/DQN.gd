class_name DQN

# The main neural network used for learning
var neural_network: NeuralNetworkAdvanced

# Hyper-parameters
var use_target_network: bool = true  # Whether to use a target network for stability
var target_neural_network: NeuralNetworkAdvanced  # Optional target network
var update_target_every_steps: int = 1000  # Frequency of target network updates

var exploration_probability: float = 1.0  # Initial exploration probability for epsilon-greedy strategy
var exploration_decreasing_decay: float = 0.01  # Rate at which exploration probability decreases
var min_exploration_probability: float = 0.05  # Minimum exploration probability
var discounted_factor: float = 0.9  # Discount factor for future rewards (gamma)
var learning_rate: float = 0.2  # Learning rate for neural network updates
var use_l2_regularization: bool = true  # Flag to use L2 regularization
var l2_regularization_strength: float = 0.001  # Strength of L2 regularization
var decay_per_steps: int = 100  # Frequency of decay for exploration probability
var print_debug_info: bool = false  # Flag to print debug information

# Exploration strategy can be "epsilon_greedy" or "softmax"
var exploration_strategy: String = "epsilon_greedy"
# Sampling strategy for replay memory can be "random", "sequential", "priority", or "time_series"
var sampling_strategy: String = "random"

# Replay memory settings
var memory_capacity: int = 300  # Maximum capacity of replay memory
var replay_memory: Array = []  # Stores past experiences for replay
var batch_size: int = 30  # Number of experiences sampled from replay memory for training

# Variables for tracking steps and learning status
var steps_completed: int = 0  # Total number of steps completed
var is_learning: bool = true  # Flag to indicate if learning is enabled
var use_replay: bool = false  # Flag to enable or disable experience replay

# Variables to store the previous state and action
var previous_state: Array = []
var previous_action: int

# Initialization function
func _init(config: Dictionary) -> void:
	# Set hyper-parameters based on the provided config dictionary
	set_config(config)
	
	# Initialize the main neural network with the specified architecture
	neural_network = NeuralNetworkAdvanced.new(config)
	
	# If target network is used, create a copy of the main network
	if use_target_network:
		target_neural_network = neural_network.copy()

# Function to set hyper-parameters from a config dictionary
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
	print_debug_info = config.get("print_debug_info", print_debug_info)
	exploration_strategy = config.get("exploration_strategy", exploration_strategy)
	sampling_strategy = config.get("sampling_strategy", sampling_strategy)
	
# Function to add a new layer to the neural network
func add_layer(nodes: int, function: Dictionary = neural_network.ACTIVATIONS.SIGMOID):
	neural_network.add_layer(nodes, function)

# Function to add an experience to the replay memory
func add_to_memory(state, action, reward, next_state, done):
	if replay_memory.size() >= memory_capacity:
		replay_memory.pop_front()  # Remove the oldest experience if memory is full
	replay_memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})

# Function to update the target network with the current weights of the main network
func update_target_network():
	if use_target_network:
		print("Updated Target Network")
		target_neural_network = neural_network.copy()

# Random sampling strategy for replay memory
func sample_random_memory() -> Array:
	var batch = []
	for i in range(min(batch_size, replay_memory.size())):
		batch.append(replay_memory.pick_random())  # Randomly select experiences
	return batch

# Sequential sampling strategy for replay memory
func sample_sequential_memory() -> Array:
	var batch = []
	var max_start_index = replay_memory.size() - batch_size
	
	# Randomly select a starting point within a valid range
	var start_index = randi() % max(1, max_start_index + 1)

	for i in range(batch_size):
		if start_index + i < replay_memory.size():
			batch.append(replay_memory[start_index + i])
	return batch

# Priority sampling strategy for replay memory
func sample_priority_memory() -> Array:
	var batch = []
	var priorities = []
	var sum_priorities = 0.0

	# Calculate priorities based on absolute TD error (Temporal Difference error)
	for experience in replay_memory:
		var max_future_q = neural_network.predict(experience["next_state"]).max()
		var td_error = abs(experience["reward"] + discounted_factor * max_future_q - neural_network.predict(experience["state"])[experience["action"]])
		priorities.append(td_error)
		sum_priorities += td_error

	# Sample experiences based on their priorities
	for n in range(batch_size):
		var rand_value = randf() * sum_priorities
		for i in range(priorities.size()):
			rand_value -= priorities[i]
			if rand_value <= 0:
				batch.append(replay_memory[i])
				break

	return batch

# Time series sampling strategy for replay memory
func sample_time_series_memory(sequence_length: int) -> Array:
	var batch = []
	var max_start_index = replay_memory.size() - sequence_length

	for i in range(min(batch_size, max_start_index)):
		var start_index = randi() % max_start_index  # Random starting point
		var sequence = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

		# Collect a sequence of experiences
		for j in range(sequence_length):
			var experience = replay_memory[start_index + j]
			sequence["state"].append(experience["state"])
			sequence["action"].append(experience["action"])
			sequence["reward"].append(experience["reward"])
			sequence["next_state"].append(experience["next_state"])
			sequence["done"].append(experience["done"])
		
		batch.append(sequence)
	return batch

# Generalized function to sample replay memory based on the chosen strategy
func sample_replay_memory(sampling_strategy: String) -> Array:
	match sampling_strategy:
		"random":
			return sample_random_memory()  # Random sampling
		"sequential":
			return sample_sequential_memory()  # Sequential sampling
		"priority":
			return sample_priority_memory()  # Priority sampling
		"time_series":
			return sample_time_series_memory(5)  # Time series sampling with a sequence length of 5
		_:
			return sample_random_memory()  # Default to random sampling

# Function to train the network on a batch of experiences
func train_batch(batch: Array):
	for experience in batch:
		# Determine if the batch contains time series data by checking the type of "state"
		if typeof(experience["state"]) == TYPE_ARRAY and experience["state"].size() > 0 and typeof(experience["state"][0]) == TYPE_ARRAY:
			# Handle time series data
			var sequence_length = experience["state"].size()
			for t in range(sequence_length):
				var current_state = experience["state"][t]
				var next_state = experience["next_state"][t]
				var reward = experience["reward"][t]
				var action = experience["action"][t]
				var done = experience["done"][t]
				
				var max_future_q: float
				if use_target_network:
					max_future_q = target_neural_network.predict(next_state).max()
				else:
					max_future_q = neural_network.predict(next_state).max()
				
				# Calculate target Q-value
				var target_q_value = reward + discounted_factor * max_future_q if not done else reward
				var target_q_values = neural_network.predict(current_state)
				target_q_values[action] = target_q_value
				neural_network.train(current_state, target_q_values)
		else:
			# Handle regular (non-time series) data
			var max_future_q: float
			if use_target_network:
				max_future_q = target_neural_network.predict(experience["next_state"]).max()
			else:
				max_future_q = neural_network.predict(experience["next_state"]).max()
			
			# Calculate target Q-value
			var target_q_value = experience["reward"] + discounted_factor * max_future_q if not experience["done"] else experience["reward"]
			var target_q_values = neural_network.predict(experience["state"])
			target_q_values[experience["action"]] = target_q_value
			neural_network.train(experience["state"], target_q_values)

# Function to calculate softmax probabilities from Q-values
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

# Function to choose an action based on softmax probabilities
func choose_action_softmax(q_values: Array) -> int:
	var probabilities = softmax(q_values)
	var cumulative_prob = 0.0
	var rand_value = randf()
	for i in range(probabilities.size()):
		cumulative_prob += probabilities[i]
		if rand_value < cumulative_prob:
			return i
	# Fallback: choose a random action if no threshold is met
	return randi() % q_values.size()

# Main training function called at each step
func train(current_states: Array, reward_of_previous_state: float, done: bool = false) -> int:
	var current_q_values = neural_network.predict(current_states)
	var current_action = choose_action(current_states)
	
	# If there is a previous state, update the network
	if previous_state.size() != 0:
		if use_replay:
			# Add experience to replay memory
			add_to_memory(previous_state, previous_action, reward_of_previous_state, current_states, done)
			# If memory is full enough, sample a batch and train
			if replay_memory.size() >= batch_size:
				var batch = sample_replay_memory(sampling_strategy)
				train_batch(batch)
		else:
			# Update the network directly without replay
			var max_future_q: float
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
	
	# Update target network periodically
	if steps_completed % update_target_every_steps == 0:
		update_target_network()

	# Decay exploration probability periodically
	if steps_completed % decay_per_steps == 0:
		if print_debug_info:
			print("Total steps completed:", steps_completed)
			print("Current exploration probability:", exploration_probability)
			print("Q-Net data:", neural_network.debug())
			print("-----------------------------------------------------------------------------------------")
	
	steps_completed += 1
	return previous_action

# Function to choose the next action based on the current state and exploration strategy
func choose_action(current_states: Array) -> int:
	var current_q_values = neural_network.predict(current_states)
	var action_to_take: int

	# Choose action based on the exploration strategy
	if exploration_strategy == "epsilon_greedy":
		if randf() < exploration_probability:
			action_to_take = randi() % current_q_values.size()  # Random action
		else:
			action_to_take = current_q_values.find(current_q_values.max())  # Greedy action
		# Decay exploration probability
		exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
		
	elif exploration_strategy == "softmax":
		action_to_take = choose_action_softmax(current_q_values)
	else:
		action_to_take = current_q_values.find(current_q_values.max())  # Default to greedy action

	return action_to_take

# Function to save the neural network state to a file
func save(path):
	neural_network.save(path)

# Function to load the neural network state from a file and configure the DQN
func load(path, config: Dictionary = {"exploration_strategy": "softmax"}):
	neural_network.load(path)
	set_config(config)
	update_target_network()
