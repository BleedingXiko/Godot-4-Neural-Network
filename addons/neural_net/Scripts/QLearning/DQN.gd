class_name DQN

var neural_network: NeuralNetworkAdvanced

# Hyper-parameters
# Optional target network
var use_target_network: bool = true
var target_neural_network: NeuralNetworkAdvanced
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
var sampling_strategy: String = "random"

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
	# Initialize the neural network with fixed architecture
	neural_network = NeuralNetworkAdvanced.new(config)
	
	# Initialize the target network if required
	if use_target_network:
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
	sampling_strategy = config.get("sampling_strategy", sampling_strategy)
	

func add_layer(nodes: int, function: Dictionary = neural_network.ACTIVATIONS.SIGMOID):
	neural_network.add_layer(nodes, function)

func add_to_memory(state, action, reward, next_state, done):
	if replay_memory.size() >= memory_capacity:
		replay_memory.pop_front()
	replay_memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})


func update_target_network():
	if use_target_network:
		print("updated Target Network")
		target_neural_network = neural_network.copy()

# Random sampling (default)
func sample_random_memory() -> Array:
	var batch = []
	for i in range(min(batch_size, replay_memory.size())):
		batch.append(replay_memory.pick_random())
	return batch

# Sequential sampling
func sample_sequential_memory() -> Array:
	var batch = []
	var max_start_index = replay_memory.size() - batch_size
	
	# Randomly select a starting point within a valid range
	var start_index = randi() % max(1, max_start_index + 1)

	for i in range(batch_size):
		if start_index + i < replay_memory.size():
			batch.append(replay_memory[start_index + i])
	return batch




# Priority sampling
func sample_priority_memory() -> Array:
	var batch = []
	var priorities = []
	var sum_priorities = 0.0

	# Calculate priorities (e.g., based on absolute TD error)
	for experience in replay_memory:
		var max_future_q = neural_network.predict(experience["next_state"]).max()
		var td_error = abs(experience["reward"] + discounted_factor * max_future_q - neural_network.predict(experience["state"])[experience["action"]])
		priorities.append(td_error)
		sum_priorities += td_error

	# Sample based on priorities
	for n in range(batch_size):
		var rand_value = randf() * sum_priorities
		for i in range(priorities.size()):
			rand_value -= priorities[i]
			if rand_value <= 0:
				batch.append(replay_memory[i])
				break

	return batch

# Time series sampling


func sample_time_series_memory(sequence_length: int) -> Array:
	var batch = []
	var max_start_index = replay_memory.size() - sequence_length

	for i in range(min(batch_size, max_start_index)):
		var start_index = randi() % max_start_index  # Random starting point
		var sequence = {"state": [], "action": [], "reward": [], "next_state": [], "done": []}

		for j in range(sequence_length):
			var experience = replay_memory[start_index + j]
			sequence["state"].append(experience["state"])
			sequence["action"].append(experience["action"])
			sequence["reward"].append(experience["reward"])
			sequence["next_state"].append(experience["next_state"])
			sequence["done"].append(experience["done"])
		
		batch.append(sequence)
	return batch


# Generalized sample memory function
func sample_replay_memory(sampling_strategy: String) -> Array:
	match sampling_strategy:
		"random":
			return sample_random_memory()  # Existing random sampling
		"sequential":
			return sample_sequential_memory()
		"priority":
			return sample_priority_memory()
		"time_series":
			return sample_time_series_memory(5)  # Example sequence length
		_:
			return sample_random_memory()  # Default to random sampling



func train_batch(batch: Array):
	for experience in batch:
		# Determine if the batch is time series by checking if the "state" is an array of arrays
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
	var rand_value = randf()
	for i in range(probabilities.size()):
		cumulative_prob += probabilities[i]
		if rand_value < cumulative_prob:
			return i
	# Fallback: choose a random action if the cumulative probability never exceeds rand_value
	return randi() % q_values.size()


func train(current_states: Array, reward_of_previous_state: float, done: bool = false) -> int:
	var current_q_values = neural_network.predict(current_states)
	var current_action = choose_action(current_states)
	
	# Handle the learning and updating process
	if previous_state.size() != 0:
		if use_replay:
			add_to_memory(previous_state, previous_action, reward_of_previous_state, current_states, done)
			if replay_memory.size() >= batch_size:
				var batch = sample_replay_memory(sampling_strategy)
				train_batch(batch)
		else:
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
