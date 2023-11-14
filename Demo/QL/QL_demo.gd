extends Node2D

var qt: QTable
var row: int = 0
var column: int = 0

var reward_states = [4, 24, 35]
var target: int = reward_states.pick_random()
var punish_states = [3, 18, 21, 28, 31]

var current_state: Array = []
var previous_reward: float = 0.0

var total_iteration_rewards: Array[float] = []
var current_iteration_rewards: float = 0.0
var done: bool = false

func _ready() -> void:
	qt = QTable.new(36 * 3, 4,2, true)
	qt.print_debug_info = true
	qt.exploration_decreasing_decay = 0.01 # Exploration decay
	qt.min_exploration_probability = 0.05 # Minimum exploration probability
	qt.discounted_factor = 0.9 # Discount factor (gamma)
	qt.learning_rate = 0.2 # Learning rate
	qt.decay_per_steps = 100


func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$Timer.wait_time = 0.5
	elif Input.is_action_just_pressed("ui_down"):
		$Timer.wait_time = 0.001
	elif Input.is_action_just_pressed("ui_up"):
		qt.load('./qnet.data')

func _on_timer_timeout():
	if done:
		reset()
		return
	current_state = [row * 6 + column, target]
	var action_to_do: int = qt.predict(current_state, previous_reward)
	
	current_iteration_rewards += previous_reward
	previous_reward = 0.0
		
	if is_out_bound(action_to_do):
		previous_reward -= 0.75
		done = true
	elif row * 6 + column in punish_states:
		previous_reward -= 0.5
		done = true
	elif (row * 6 + column) == target:
		previous_reward += 1.0
		done = true
	else:
		previous_reward -= 0.05
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))
	$lr.text = str(qt.exploration_probability)
	$target.text = str(target)


func is_out_bound(action: int) -> bool:
	var _column := column
	var _row := row
	match action:
		0:
			_column -= 1
		1:
			_row += 1
		2:
			_column += 1
		3:
			_row -= 1
	if _column < 0 or _row < 0 or _column > 5 or _row > 5:
		return true
	else:
		column = _column
		row = _row
		return false

func reset():
	target = reward_states.pick_random()
	row = randi_range(1, 5)
	column = randi_range(1, 5)
	done = false
	total_iteration_rewards.append(current_iteration_rewards)
	current_iteration_rewards = 0.0
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))

