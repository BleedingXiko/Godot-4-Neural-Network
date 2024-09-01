extends Node2D

const GRID_SIZE = 4  # 4x4 grid for 2048
const EMPTY = 0.0

var grid: Array = []
var previous_grid: Array = []
var previous_action: int = -1  # No action taken initially

var dqn: DDQN  # Assuming you already have a DQN class implemented

# Neural network configuration
var config = {
	"print_debug_info": false,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.001,
	"min_exploration_probability": 0.1,
	"exploration_strategy": "epsilon_greedy",
	"sampling_strategy": "sequential",
	"discounted_factor": 0.99,
	"decay_per_steps": 100,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 200,
	"memory_capacity": 2048,
	"batch_size": 128,
	"learning_rate": 0.001,
	"use_l2_regularization": true,
	"l2_regularization_strength": 1,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7,
	"early_stopping": false,
	"patience": 20,
	"save_path": "res://dqn_2048.data",
	"smoothing_window": 10,
	"check_frequency": 5,
	"minimum_epochs": 100,
	"improvement_threshold": 0.0001,
	"use_gradient_clipping": true,
	"gradient_clip_value": 1.0,
	"initialization_type": "xavier"
}

func _ready() -> void:
	# Initialize grid and DQN
	initialize_grid()
	dqn = DDQN.new(config)
	dqn.add_layer(GRID_SIZE * GRID_SIZE * 2 + 1)  # Input size: current grid, previous grid, and previous action
	dqn.add_layer(86, dqn.neural_network.ACTIVATIONS.RELU)  # Hidden layer
	dqn.add_layer(4, dqn.neural_network.ACTIVATIONS.SIGMOID)  # Output layer: 4 possible actions (up, down, left, right)

	# Optionally load a pre-trained model
	#dqn.load("user://dqn_2048.data")

	# Train the agent
	for i in range(100):  # Number of episodes
		run_training_episode()
		if i % 10 == 0:
			dqn.save("user://dqn_2048.data")

	# After training, print the board and watch it play
	print("Training complete!")
	visualize_gameplay()

func initialize_grid():
	grid = []
	previous_grid = []
	for x in range(GRID_SIZE):
		var row: Array = []
		var prev_row: Array = []
		for y in range(GRID_SIZE):
			row.append(EMPTY)
			prev_row.append(EMPTY)
		grid.append(row)
		previous_grid.append(prev_row)
	previous_action = -1

func add_random_tile():
	var empty_positions: Array = []
	for x in range(GRID_SIZE):
		for y in range(GRID_SIZE):
			if grid[x][y] == EMPTY:
				empty_positions.append(Vector2i(x, y))

	if empty_positions.size() > 0:
		var position = empty_positions[randi() % empty_positions.size()]
		grid[position.x][position.y] = 2 if randf() < 0.9 else 4

func move_tiles(direction: Vector2i) -> bool:
	var moved = false

	# We need to move and merge tiles in the direction specified.
	# We will perform operations in a way that ensures we handle merges correctly.
	
	# Horizontal movement (Left or Right)
	if direction.x != 0:
		for y in range(GRID_SIZE):
			var row = []
			# Extract the row
			for x in range(GRID_SIZE):
				row.append(grid[x][y])

			# Process the row
			var processed_row = process_line(row, direction.x == 1)  # Right direction = 1
			for x in range(GRID_SIZE):
				if grid[x][y] != processed_row[x]:
					moved = true
				grid[x][y] = processed_row[x]
	
	# Vertical movement (Up or Down)
	if direction.y != 0:
		for x in range(GRID_SIZE):
			var column = []
			# Extract the column
			for y in range(GRID_SIZE):
				column.append(grid[x][y])

			# Process the column
			var processed_column = process_line(column, direction.y == 1)  # Down direction = 1
			for y in range(GRID_SIZE):
				if grid[x][y] != processed_column[y]:
					moved = true
				grid[x][y] = processed_column[y]

	return moved

# Helper function to process each row or column
# `reverse` indicates whether to process in the reverse direction (right or down)
func process_line(line: Array, reverse: bool) -> Array:
	if reverse:
		line.reverse()

	# Shift non-zero tiles to the front
	var new_line = []
	for value in line:
		if value != EMPTY:
			new_line.append(value)
	
	# Merge tiles
	for i in range(new_line.size() - 1):
		if new_line[i] == new_line[i + 1]:
			new_line[i] *= 2
			new_line[i + 1] = EMPTY
	
	# Remove the merged tiles (empty spaces)
	var final_line = []
	for value in new_line:
		if value != EMPTY:
			final_line.append(value)
	
	# Pad the remaining empty spaces to the right
	while final_line.size() < GRID_SIZE:
		final_line.append(EMPTY)
	
	if reverse:
		final_line.reverse()

	return final_line

func run_training_episode():
	initialize_grid()
	add_random_tile()
	add_random_tile()

	var done = false
	while not done:
		# Capture current state
		var state = get_state()

		# Choose an action
		var action = dqn.choose_action(state)

		# Perform action and update grid
		var direction = get_direction(action)
		var moved = move_tiles(direction)
		if moved:
			add_random_tile()
		done = check_game_over()

		# Calculate reward
		var reward = calculate_reward()

		# Train the DQN
		dqn.train(state, reward, done)

		# Update previous state and action
		if moved:
			update_previous_state_and_action(action)

func get_state() -> Array:
	var state: Array = []

	# Flatten the current grid and previous grid and append them to state
	for x in range(GRID_SIZE):
		for y in range(GRID_SIZE):
			state.append(grid[x][y])
			state.append(previous_grid[x][y])

	# Append the previous action
	state.append(previous_action)

	return state

func get_direction(action: int) -> Vector2i:
	# Convert action index to a direction vector
	match action:
		0:
			return Vector2i(0, -1)  # Up
		1:
			return Vector2i(0, 1)  # Down
		2:
			return Vector2i(-1, 0)  # Left
		3:
			return Vector2i(1, 0)  # Right
	return Vector2i(0, 0)  # No move

func calculate_reward() -> float:
	# Reward can be calculated based on the increase in the sum of tile values
	# or the appearance of new higher-value tiles, or simply based on whether a move was successful
	var current_score = 0.0
	for row in grid:
		for value in row:
			current_score += value
	return current_score  # Example reward is just the current score

func check_game_over() -> bool:
	# Check if there are no valid moves left
	for x in range(GRID_SIZE):
		for y in range(GRID_SIZE):
			if grid[x][y] == EMPTY:
				return false  # If there's an empty space, the game is not over
			if x < GRID_SIZE - 1 and grid[x][y] == grid[x + 1][y]:
				return false  # Adjacent horizontal match
			if y < GRID_SIZE - 1 and grid[x][y] == grid[x][y + 1]:
				return false  # Adjacent vertical match
	return true  # No valid moves left

func update_previous_state_and_action(action: int):
	# Copy the current grid to the previous grid and update the previous action
	previous_grid = grid.duplicate()
	previous_action = action

func print_board():
	print("-----------------")
	for row in grid:
		var row_string = "|"
		for value in row:
			row_string += "%4d |" % value if value > 0 else "     |"
		print(row_string)
	print("-----------------")

func visualize_gameplay():
	initialize_grid()
	add_random_tile()
	add_random_tile()

	var done = false
	while not done:
		print_board()
		var state = get_state()
		var action = dqn.choose_action(state)
		var direction = get_direction(action)
		var moved = move_tiles(direction)
		if moved:
			add_random_tile()
		done = check_game_over()
		await get_tree().create_timer(0.5).timeout  # Slow down to watch the game unfold
	print("Game Over!")
