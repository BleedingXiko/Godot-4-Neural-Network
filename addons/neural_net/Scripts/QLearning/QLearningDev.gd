class_name QLearning

# Observation Spaces are the possible states the agent can be in
# Action Spaces are the possible actions the agent can take
var observation_space: int
var action_spaces: int

# The table that contains the value for each cell in the QLearning algorithm
var QTable: Dictionary

# Hyper-parameters
var exploration_probability: float = 1.0
var exploration_decreasing_decay: float = 0.01
var min_exploration_probability: float = 0.01
var discounted_factor: float = 0.9
var learning_rate: float = 0.2
var decay_per_steps: int = 100
var steps_completed: int = 0

# States
var previous_state: Array = []
var current_state: Array = []
var previous_action: int

var done: bool = false
var is_learning: bool = true
var print_debug_info: bool = false

func _init(n_observations: int, n_action_spaces: int, _is_learning: bool = true) -> void:
    observation_space = n_observations
    action_spaces = n_action_spaces
    is_learning = _is_learning

    QTable = {}

func max_from_row(dictionary: Dictionary, row_key: Array) -> float:
    var max_value: float = -INFINITY

    for key in dictionary.keys():
        if key[0] == row_key:
            var value = dictionary[key]
            if value > max_value:
                max_value = value

    return max_value

func predict(current_state: Array, reward_of_previous_state: float) -> int:
    if is_learning:
        if previous_state.size() > 0:
            var prev_key = [previous_state, previous_action]
            if QTable.has(prev_key):
                QTable[prev_key] = (1 - learning_rate) * QTable[prev_key] + \
                learning_rate * (reward_of_previous_state + discounted_factor * max_from_row(QTable, current_state))
            else:
                QTable[prev_key] = reward_of_previous_state

    var action_to_take: int

    if randf() < exploration_probability and is_learning:
        action_to_take = randi_range(0, action_spaces - 1)
    else:
        var current_state_key = [current_state, action_to_take]
        if QTable.has(current_state_key):
            action_to_take = QTable[current_state_key]
        else:
            action_to_take = 0

    if is_learning:
        previous_state = current_state
        previous_action = action_to_take

        if steps_completed != 0 and steps_completed % decay_per_steps == 0:
            exploration_probability = maxf(min_exploration_probability, exploration_probability - exploration_decreasing_decay)

    if print_debug_info and steps_completed % decay_per_steps == 0:
        print("Total steps completed:", steps_completed)
        print("Current exploration probability:", exploration_probability)
        print("Q-Table data:", QTable)
        print("-----------------------------------------------------------------------------------------")

    steps_completed += 1
    return action_to_take