class_name QTable

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int
var action_spaces: int

# The table that contains the value for each cell in the QLearning algorithm
var Table: Matrix

# Hyper-parameters
var exploration_probability: float = 1.0 # Probability of exploring
var exploration_decreasing_decay: float = 0.01 # Exploration decay
var min_exploration_probability: float = 0.05 # Minimum exploration probability
var discounted_factor: float = 0.9 # Discount factor (gamma)
var learning_rate: float = 0.2 # Learning rate
var decay_per_steps: int = 100
var steps_completed: int = 0

# States
var previous_state: int = -100 # Previous state
var previous_action: int # Previous action taken

var MAX_STATE_VALUE: int = 2 # Insures each composite state will be unique and needs to be within the bounds of the table
var is_learning: bool = true
var print_debug_info: bool = false

func _init(n_observations: int, n_action_spaces: int, config: Dictionary) -> void:
	observation_space = n_observations
	action_spaces = n_action_spaces
	
	is_learning = config.get("is_learning", is_learning)
	MAX_STATE_VALUE = config.get("MAX_STATE_VALUE", MAX_STATE_VALUE)
	exploration_decreasing_decay = config.get("exploration_decreasing_decay", exploration_decreasing_decay)
	min_exploration_probability = config.get("min_exploration_probability", min_exploration_probability)
	discounted_factor = config.get("discounted_factor", discounted_factor)
	learning_rate = config.get("learning_rate", learning_rate)
	decay_per_steps = config.get("decay_per_steps", decay_per_steps)
	print_debug_info = config.get("print_debug_info", print_debug_info)
	Table = Matrix.new(observation_space, action_spaces)
	#QTable = Matrix.rand(Matrix.new(observation_space, action_spaces))
	# Optionally initialize QTable with random values

func predict(current_states: Array, reward_of_previous_state: float) -> int:
	# Create a composite state from current states
	var chosen_state = create_composite_state(current_states)

	# Update Q-Table for the previous state-action pair
	if is_learning and previous_state != -100:
		var old_value = Table.data[previous_state][previous_action]
		var max_future_q = Table.max_from_row(chosen_state)
		var new_value = (1 - learning_rate) * old_value + learning_rate * (reward_of_previous_state + discounted_factor * max_future_q)
		Table.data[previous_state][previous_action] = new_value

	# Action selection based on exploration-exploitation trade-off
	var action_to_take: int
	if randf() < exploration_probability:
		action_to_take = randi() % action_spaces
	else:
		action_to_take = Table.index_of_max_from_row(chosen_state)

	# Update exploration probability and other states
	if is_learning:
		previous_state = chosen_state
		previous_action = action_to_take
		if steps_completed % decay_per_steps == 0:
			exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)

	if print_debug_info and steps_completed % decay_per_steps == 0:
		print("Total steps completed:", steps_completed)
		print("Current exploration probability:", exploration_probability)
		print("Q-Table data:", Table.data)
		print("-----------------------------------------------------------------------------------------")

	steps_completed += 1
	return action_to_take

func create_composite_state(current_states: Array) -> int:
	var composite_state = 0
	var multiplier = 1
	for state in current_states:
		composite_state += state * multiplier
		multiplier *= MAX_STATE_VALUE # Define MAX_STATE_VALUE based on your state ranges
	return composite_state



func save(path):
	var file = FileAccess.open(path, FileAccess.WRITE)
	var data = Table.save()
	file.store_var(data)
	file.close()

func load(path):
	var file = FileAccess.open(path, FileAccess.READ)
	var data = file.get_var()
	Table = Matrix.load(data)
	is_learning = false
	exploration_probability = min_exploration_probability
	file.close()
