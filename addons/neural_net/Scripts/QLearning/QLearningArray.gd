class_name QLearningArray

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int
var action_spaces: int

# The table that contains the value for each cell in the QLearning algorithm
var QTable: Matrix

# Hyper-parameters
var exploration_probability: float = 1.0 # The probability that the agent will either explore or exploit the QTable
var exploration_decreasing_decay: float = 0.01 # The exploration decreasing decay for exponential decreasing
var min_exploration_probability: float = 0.01 # The least value that the exploration_probability can fall to
var discounted_factor: float = 0.9 # Basically the gamma
var learning_rate: float = 0.2 # How fast the agent learns
var decay_per_steps: int = 100
var steps_completed: int = 0

# States
var previous_state: int = -100 # To be used in the algorithms
var current_states: Array # To store multiple current states
var previous_action: int # To be used in the algorithm

var is_learning: bool = true
var print_debug_info: bool = false

func _init(n_observations: int, n_action_spaces: int, _is_learning: bool = true) -> void:
	observation_space = n_observations
	action_spaces = n_action_spaces
	is_learning = _is_learning
	QTable = Matrix.new(observation_space, action_spaces)
	#QTable = QTable.rand(QTable)

func predict(current_states: Array, reward_of_previous_state: float) -> int:
	var action_to_take: int

	# Update Q values for each state in current_states
	for state in current_states:
		if is_learning and previous_state != -100:
			# Update the Q value for the previous state and action
			QTable.data[previous_state][previous_action] = (1 - learning_rate) * QTable.data[previous_state][previous_action] + \
			learning_rate * (reward_of_previous_state + discounted_factor * QTable.max_from_row(state))

	# Select an action based on one of the states (e.g., the first or last)
	# Modify this part as needed, e.g., use a different criterion for choosing the state
	var chosen_state: int
	
	if current_states.size() > 1:
		var total_state: int = 0
		for state in current_states:
			total_state += state
		chosen_state = round(total_state / current_states.size())
	
	#if current states is only an array with one value treat it like a regular Qlearning int state
	else:
		chosen_state = current_states[0]
#	var chosen_state: int = current_states.pick_random()
 # or current_states.front() or any other criterion
	
	if randf() < exploration_probability:
		action_to_take = randi() % action_spaces
	else:
		action_to_take = QTable.index_of_max_from_row(chosen_state)
		
	# Update exploration probability and logging
	if is_learning:
		previous_state = chosen_state
		previous_action = action_to_take
		if steps_completed % decay_per_steps == 0:
			exploration_probability = max(min_exploration_probability, exploration_probability - exploration_decreasing_decay)

	if print_debug_info and steps_completed % decay_per_steps == 0:
		print("Total steps completed:", steps_completed)
		print("Current exploration probability:", exploration_probability)
		print("Q-Table data:", QTable.data)
		print("-----------------------------------------------------------------------------------------")
	steps_completed += 1
	return action_to_take

func save(path):
	var file = FileAccess.open(path, FileAccess.WRITE)
	var data = QTable.save()
	file.store_var(data)
	file.close()

func load(path):
	var file = FileAccess.open(path, FileAccess.READ)
	var data = file.get_var()
	QTable.data = data
	is_learning = false
	exploration_probability = 0.05
	file.close()
