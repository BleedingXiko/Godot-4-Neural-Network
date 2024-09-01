extends Node2D

var dqn: DDQN
var row: int = 0
var column: int = 0

var reward_states = [4, 24, 35]
var target: int = reward_states.pick_random()
var punish_states = [3, 18, 21, 28, 31]

var grid_size: int = 6
var grid: Array = []

var previous_reward: float = 0.0
var total_iteration_rewards: Array[float] = []
var current_iteration_rewards: float = 0.0
var done: bool = false

var dqn_config = {
	"print_debug_info": false,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.001,
	"min_exploration_probability": 0.15,
	"exploration_strategy": "softmax",  # Changed to epsilon-greedy
	"sampling_strategy": "sequential",
	"discounted_factor": 0.99,
	"decay_per_steps": 250,
	"use_replay": false,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 500,
	"memory_capacity": 1024,
	"batch_size": 128,
	"learning_rate": 0.1,
	"use_l2_regularization": true,
	"l2_regularization_strength": 0.1,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7,
	"early_stopping": false,
	"patience": 15,
	"save_path": "res://dqn_snake.data",
	"smoothing_window": 100,
	"check_frequency": 10,
	"minimum_epochs": 200,
	"improvement_threshold": 0.00005,
	"use_gradient_clipping": true,
	"gradient_clip_value": 1.0,
	"initialization_type": "xavier" 
}

func _ready() -> void:
	dqn = DDQN.new(dqn_config)
	dqn.add_layer(grid_size * grid_size)  # Input layer matches the entire grid
	dqn.add_layer(22, dqn.neural_network.ACTIVATIONS.RELU)  # Hidden layer with more neurons to handle spatial complexity
	dqn.add_layer(4, dqn.neural_network.ACTIVATIONS.SIGMOID)  # Output layer: 4 possible actions (up, down, left, right)
	reset()

func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$Timer.wait_time = 0.5
	elif Input.is_action_just_pressed("ui_down"):
		$Timer.wait_time = 0.001

func _on_timer_timeout():
	update_grid()  # Refresh grid representation based on current positions
	var action_to_do: int = dqn.choose_action(grid)
	if done:
		reset()

	current_iteration_rewards += previous_reward
	previous_reward = 0.0

	if is_out_bound(action_to_do):
		previous_reward -= 0.75
		done = true
	elif row * grid_size + column in punish_states:
		previous_reward -= 0.5
		done = true
	elif (row * grid_size + column) == target:
		previous_reward += 1.0
		done = true
		target = reward_states.pick_random()
	else:
		previous_reward -= 0.05

	# Update the DQN with the new state, action, and reward
	dqn.train(grid, previous_reward, done)

	# Update the player's position visually
	$player.position = Vector2(96 * column + 16, 96 * (grid_size - 1 - row) + 16)  # Aligns correctly with the grid
	$lr.text = str(dqn.exploration_probability)
	$target.text = str(target)

func update_grid():
	grid = []
	for i in range(grid_size * grid_size):
		grid.append(0)  # Initialize the grid with empty cells (0)
	
	# Correctly align the player's position (flip row index)
	grid[(grid_size - 1 - row) * grid_size + column] = 1  # Player's position is marked as 1
	
	# Correctly align the target position (flip row index)
	var target_row = int(target / grid_size)
	var target_col = target % grid_size
	grid[(grid_size - 1 - target_row) * grid_size + target_col] = 2  # Target position is marked as 2
	
	# Correctly align punishment positions (flip row index)
	for punish_state in punish_states:
		var punish_row = int(punish_state / grid_size)
		var punish_col = punish_state % grid_size
		grid[(grid_size - 1 - punish_row) * grid_size + punish_col] = -1  # Punishment positions are marked as -1

	# Display the grid in the console to visualize what the AI sees
	print("AI Grid View:")
	for i in range(grid_size):
		var row_str = ""
		for j in range(grid_size):
			row_str += str(grid[i * grid_size + j]) + " "
		print(row_str)
	print("\n")



func is_out_bound(action: int) -> bool:
	var _column := column
	var _row := row
	match action:
		0: _column -= 1  # Left
		1: _row += 1     # Down
		2: _column += 1  # Right
		3: _row -= 1     # Up
	if _column < 0 or _row < 0 or _column >= grid_size or _row >= grid_size:
		return true
	else:
		column = _column
		row = _row
		return false

func reset():
	target = reward_states.pick_random()
	row = randi_range(0, grid_size - 1)
	column = randi_range(0, grid_size - 1)
	done = false
	total_iteration_rewards.append(current_iteration_rewards)
	current_iteration_rewards = 0.0
	$player.position = Vector2(96 * column + 16, 96 * (grid_size - 1 - row) + 16)  # Correctly aligns with the grid
	update_grid()

func _on_save_pressed():
	dqn.save('user://dqn.data')

func _on_load_pressed():
	dqn.load('user://dqn.data', dqn_config)
