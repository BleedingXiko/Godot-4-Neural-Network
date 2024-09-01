extends Node2D


const GRID_SIZE = Vector2i(4, 4)  # The size of the grid (4x4)
const EMPTY = 0.05
const SNAKE_HEAD = 0.9
const SNAKE_BODY = 0.7
const FOOD = 0.3
const PREV_SNAKE_HEAD = -0.9
const PREV_SNAKE_BODY = -0.7
const PREV_FOOD = -0.3

var grid: Array = []
var prev_grid: Array = []  # Add previous grid state
var snake: Array = []
var direction: Vector2i = Vector2i(1, 0)  # Start moving right
var food_position: Vector2i = Vector2i.ZERO
var game_over: bool = false
var previous_action: int = -1  # Initialize with an invalid action

var ppo: PPO
var training_done: bool = false

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

# PPO configurations
var actor_config = {
	"learning_rate": 0.001,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.001
}

var critic_config = {
	"learning_rate": 0.001,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.01
}

var training_config = {
	"gamma": 0.95,
	"epsilon_clip": 0.2,
	"update_steps": 40,
	"max_memory_size": 500,
	"batch_size": 32,
	"lambda": 0.90,
	"entropy_beta": 0.03,
	"initial_learning_rate": 0.0001,
	"min_learning_rate": 0.00001,
	"decay_rate": 0.95,
	"clip_value": 0.2,
	"target_network_update_steps": 2000,  # Steps to update target network
	"learning_rate_schedule_type": "constant",  # Type of learning rate scheduling
	"accuracy_threshold": 0.90,  # Threshold for accuracy-based learning rate adjustment
	"use_gae": true,
	"use_entropy": true,
	"use_target_network": true,
	"use_gradient_clipping": false,
	"use_learning_rate_scheduling": true
}


func _ready():
	initialize_grid()
	place_snake()
	spawn_food()

	# Initialize PPO with the provided configurations
	ppo = PPO.new(actor_config, critic_config)
	ppo.set_config(training_config)
	ppo.actor.add_layer(5 * 5 * 2)  # Adjust for current + previous state
	ppo.actor.add_layer(43, ACTIVATIONS.TANH)
	ppo.actor.add_layer(4, ACTIVATIONS.SIGMOID)  # Using SIGMOID activation for 4 possible actions (up, down, left, right)

	ppo.critic.add_layer(5 * 5 * 2)  # Adjust for current + previous state
	ppo.critic.add_layer(32, ACTIVATIONS.TANH)
	ppo.critic.add_layer(42, ACTIVATIONS.TANH)
	ppo.critic.add_layer(1, ACTIVATIONS.LINEAR)
	
	#ppo.load("res://ppo_snake.data")
	# Train the agent
	for i in range(250):  # Adjust the number of training episodes as needed
		print("Training episode:", i + 1)
		run_training_episode()

	training_done = true
	print("Training complete!")
	ppo.save("res://ppo_snake.data")

	# Now, visualize the trained agent playing the game by printing the board
	visualize_gameplay()

func initialize_grid():
	grid = []
	prev_grid = []  # Initialize previous grid
	for x in range(GRID_SIZE.x):
		var row: Array = []
		var prev_row: Array = []
		for y in range(GRID_SIZE.y):
			row.append(EMPTY)
			prev_row.append(EMPTY)
		grid.append(row)
		prev_grid.append(prev_row)
	previous_action = -1

func place_snake():
	# Calculate the center of the grid
	var center_x = GRID_SIZE.x / 2
	var center_y = GRID_SIZE.y / 2
	
	# Start the snake at the center of the grid
	snake = [Vector2i(center_x, center_y)]
	grid[snake[0].x][snake[0].y] = SNAKE_HEAD

func spawn_food():
	var empty_positions: Array = []

	# Collect all empty positions on the grid
	for x in range(GRID_SIZE.x):
		for y in range(GRID_SIZE.y):
			if grid[x][y] == EMPTY:
				empty_positions.append(Vector2i(x, y))

	if empty_positions.size() > 0:
		# Randomly select one of the empty positions for food
		food_position = empty_positions[randi() % empty_positions.size()]
		grid[food_position.x][food_position.y] = FOOD
	else:
		print("No empty position available for spawning food!")

func move_snake():
	if game_over:
		return

	# Save the current grid as the previous grid before moving the snake
	prev_grid = grid.duplicate(true)
	for x in range(GRID_SIZE.x):
		for y in range(GRID_SIZE.y):
			# Encode previous frame information
			if prev_grid[x][y] == SNAKE_HEAD:
				prev_grid[x][y] = PREV_SNAKE_HEAD
			elif prev_grid[x][y] == SNAKE_BODY:
				prev_grid[x][y] = PREV_SNAKE_BODY
			elif prev_grid[x][y] == FOOD:
				prev_grid[x][y] = PREV_FOOD

	var new_head = snake[0] + direction

	# Check if the new head is out of bounds or hits the body
	if new_head.x < 0 or new_head.x >= GRID_SIZE.x or new_head.y < 0 or new_head.y >= GRID_SIZE.y or grid[new_head.x][new_head.y] == SNAKE_BODY:
		game_over = true
		print("Game Over")
		return

	# Check if the snake has eaten the food
	if new_head == food_position:
		snake.insert(0, new_head)  # Grow the snake by adding the new head
		spawn_food()  # Respawn food after it's eaten
	else:
		# Move the snake by adding the new head
		snake.insert(0, new_head)

		# Remove the tail to simulate movement only if the snake hasn't eaten the food
		var tail = snake.pop_back()
		grid[tail.x][tail.y] = EMPTY  # Clear the tail position

	# Update the grid to reflect the new snake position
	grid[new_head.x][new_head.y] = SNAKE_HEAD  # Mark the new head
	for i in range(1, snake.size()):
		var segment = snake[i]
		grid[segment.x][segment.y] = SNAKE_BODY  # Mark the body

func run_training_episode():
	initialize_grid()
	place_snake()
	spawn_food()
	game_over = false  # Ensure game_over is reset
	
	while not game_over:
		var state = get_state()

		# Use DQN to predict the action to take
		var action = ppo.get_action(state)  # Predict the action based on the current state
		
		apply_action(action)  # Store the action taken as an integer
		
		move_snake()

		# Calculate the reward after moving the snake
		var reward = 10 if snake[0] == food_position else -100 if game_over else 0.1
		var next_state = get_state()
		# Store the experience and train the DQN
		ppo.keep(state, action, reward, next_state, game_over)

func apply_action(action: int):
	var proposed_direction: Vector2i
	previous_action = action

	match action:
		0: proposed_direction = Vector2i(1, 0)  # Right
		1: proposed_direction = Vector2i(-1, 0)  # Left
		2: proposed_direction = Vector2i(0, -1)  # Up
		3: proposed_direction = Vector2i(0, 1)  # Down

	# Prevent the snake from moving directly into its own body by checking the opposite direction
	if snake.size() > 1:
		var opposite_direction = snake[0] - snake[1]
		if proposed_direction != opposite_direction:
			direction = proposed_direction
	else:
		direction = proposed_direction

func get_state() -> Array:
	var state: Array = []
	
	# Append previous grid state
	for y in range(GRID_SIZE.y):
		for x in range(GRID_SIZE.x):
			state.append(prev_grid[x][y])

	# Append the action taken as a one-hot vector
	state.append(previous_action)

	# Append current grid state
	for y in range(GRID_SIZE.y):
		for x in range(GRID_SIZE.x):
			state.append(grid[x][y])
	
	return state

func visualize_gameplay():
	if not training_done:
		return
	
	initialize_grid()
	place_snake()
	spawn_food()
	game_over = false  # Reset game_over before starting

	while not game_over:
		var state = get_state()
		var action = ppo.get_action(state)
		apply_action(action)
		move_snake()

		print_board()

		await get_tree().create_timer(0.5).timeout  # Slow down the game so you can watch
		
		# Reset the game if it ends and keep playing
		if game_over:
			game_over = false
			initialize_grid()
			place_snake()
			spawn_food()

			print_board()

func print_board():
	print("Current State | Previous State | Action Taken: %d" % previous_action)
	for y in range(GRID_SIZE.y):
		var current_row = ""
		var prev_row = ""
		for x in range(GRID_SIZE.x):
			# Current state
			match grid[x][y]:
				EMPTY:
					current_row += ". "
				SNAKE_HEAD:
					if Vector2i(x, y) == snake[0]:
						current_row += "H "  # Mark the current head with 'H'
				SNAKE_BODY:
					current_row += "S "  # Mark the rest of the snake's body with 'S'
				FOOD:
					current_row += "F "
			
			# Previous state
			match prev_grid[x][y]:
				EMPTY:
					prev_row += ". "
				PREV_SNAKE_HEAD:
					prev_row += "h "  # Mark the previous head with 'h'
				PREV_SNAKE_BODY:
					prev_row += "s "  # Mark the previous snake's body with 's'
				PREV_FOOD:
					prev_row += "f "

		print(current_row + " | " + prev_row)
	print("\n")
