class_name QNetwork

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

func _init(n_features: int, n_nodes: Array[int], n_action_spaces: int, hidden: Dictionary, output: Dictionary,_use_replay: bool = false, _is_learning: bool = true) -> void:
	observation_space = n_features
	action_spaces = n_action_spaces
	is_learning = _is_learning
	use_replay = _use_replay

	# Initialize the neural network
	neural_network = NeuralNetworkAdvanced.new(n_features, n_nodes, action_spaces, hidden, output)
	neural_network.learning_rate = learning_rate

func add_to_memory(state, action, reward, next_state, done):
	if replay_memory.size() >= memory_capacity:
		replay_memory.pop_front()
	replay_memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})

func sample_memory():
	var batch = []
	for i in range(min(batch_size, replay_memory.size())):
		var random_index = randi() % replay_memory.size()
		batch.append(replay_memory[random_index])
	return batch

func train_batch(batch):
	for experience in batch:
		var current_q_values = neural_network.predict(experience["state"])
		var max_future_q = neural_network.predict(experience["next_state"]).max()
		var target_q_value = experience["reward"] + discounted_factor * max_future_q if not experience["done"] else experience["reward"]
		var target_q_values = neural_network.predict(experience["state"])
		target_q_values[experience["action"]] = target_q_value
		neural_network.train(experience["state"], target_q_values)

func predict(current_states: Array, reward_of_previous_state: float) -> int:
	var current_q_values = neural_network.predict(current_states)
	
	if is_learning and previous_state.size() != 0:
		#var old_q_value = neural_network.predict(previous_state)[previous_action]
		var max_future_q = current_q_values.max()
		var target_q_value = reward_of_previous_state + discounted_factor * max_future_q
		var target_q_values = neural_network.predict(previous_state)
		target_q_values[previous_action] = target_q_value
		neural_network.train(previous_state, target_q_values)
		
		if use_replay:
			add_to_memory(previous_state, previous_action, reward_of_previous_state, current_states, false) # 'false' for 'done' flag; update as necessary
			if replay_memory.size() >= batch_size:
				var batch = sample_memory()
				train_batch(batch)

	var action_to_take: int
	if randf() < exploration_probability:
		action_to_take = randi() % action_spaces
	else:
		action_to_take = current_q_values.find(current_q_values.max())

	if is_learning:
		previous_state = current_states
		previous_action = action_to_take

	if steps_completed % decay_per_steps == 0:
		exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)
	
	steps_completed += 1
	return action_to_take
