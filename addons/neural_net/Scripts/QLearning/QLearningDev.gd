class_name QLearningDev

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int
var action_spaces: int

# The table that contains the value for each cell in the QLearning algorithm
var QTable: Matrix

# Hyper-parameters
var exploration_probability: float = 1.0 # Probability of exploring
var exploration_decreasing_decay: float = 0.01 # Exploration decay
var min_exploration_probability: float = 0.01 # Minimum exploration probability
var discounted_factor: float = 0.9 # Discount factor (gamma)
var learning_rate: float = 0.2 # Learning rate
var decay_per_steps: int = 100
var steps_completed: int = 0
var MAX_STATE_VALUE: int = 36 # Defualt needs to changed based off the max value of the current state

# States
var previous_state: int = -100 # Previous state
var previous_action: int # Previous action taken

var is_learning: bool = true
var print_debug_info: bool = false

func _init(n_observations: int, n_action_spaces: int, max_state: int, _is_learning: bool = true) -> void:
    observation_space = n_observations
    action_spaces = n_action_spaces
    is_learning = _is_learning
    MAX_STATE_VALUE = max_state
    QTable = Matrix.new(observation_space, action_spaces)
    # Optionally initialize QTable with random values

func predict(current_states: Array, reward_of_previous_state: float) -> int:
    # Create a composite state from current states
    var chosen_state = create_composite_state(current_states)

    # Update Q-Table for the previous state-action pair
    if is_learning and previous_state != -100:
        var old_value = QTable.data[previous_state][previous_action]
        var max_future_q = QTable.max_from_row(chosen_state)
        var new_value = (1 - learning_rate) * old_value + learning_rate * (reward_of_previous_state + discounted_factor * max_future_q)
        QTable.data[previous_state][previous_action] = new_value

    # Action selection based on exploration-exploitation trade-off
    var action_to_take: int
    if randf() < exploration_probability:
        action_to_take = randi() % action_spaces
    else:
        action_to_take = QTable.index_of_max_from_row(chosen_state)

    # Update exploration probability and other states
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

func create_composite_state(current_states: Array) -> int:
    var composite_state = 0
    var multiplier = 1
    for state in current_states:
        composite_state += state * multiplier
        multiplier *= MAX_STATE_VALUE # Define MAX_STATE_VALUE based on your state ranges
    return composite_state



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
	exploration_probability = 0.5
	file.close()
