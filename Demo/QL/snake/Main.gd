extends Node2D

const GRID_SIZE = Vector2i(4, 4)  # The size of the grid (10x10)
const EMPTY = 0
const SNAKE_HEAD = 1
const SNAKE_BODY = -1
const FOOD = 2

var grid: Array = []
var snake: Array = []
var direction: Vector2i = Vector2i(1, 0)  # Start moving right
var food_position: Vector2i = Vector2i.ZERO
var game_over: bool = false

var dqn: QNetwork
var training_done: bool = false

# DQN configurations
var dqn_config = {
	"print_debug_info": false,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.005,
	"min_exploration_probability": 0.05,
	"exploration_strategy": "softmax",
	"discounted_factor": 0.95,
	"decay_per_steps": 250,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 1000,
	"memory_capacity": 2048,
	"batch_size": 64,
	"learning_rate": 0.0001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

func _ready():
	initialize_grid()
	place_snake()
	spawn_food()

	# Initialize DQN with the provided configurations
	dqn = QNetwork.new(dqn_config)
	dqn.add_layer(4 * 4, dqn.neural_network.ACTIVATIONS.RELU)
	dqn.add_layer(24, dqn.neural_network.ACTIVATIONS.RELU)
	dqn.add_layer(32, dqn.neural_network.ACTIVATIONS.RELU)
	dqn.add_layer(4, dqn.neural_network.ACTIVATIONS.SIGMOID)  # 4 possible actions (up, down, left, right)
	
	#dqn.load("res://dqn_snake.data", dqn_config)

	# Train the agent
	for i in range(5000):  # Adjust the number of training episodes as needed
		print("Training episode:", i + 1)
		run_training_episode()

	training_done = true
	print("Training complete!")
	
	dqn.save("res://dqn_snake.data")

	# Now, visualize the trained agent playing the game by printing the board
	visualize_gameplay()

func initialize_grid():
	grid = []
	for x in range(GRID_SIZE.x):
		var row: Array = []
		for y in range(GRID_SIZE.y):
			row.append(EMPTY)
		grid.append(row)

func place_snake():
	# Calculate the center of the grid
	var center_x = GRID_SIZE.x / 2
	var center_y = GRID_SIZE.y / 2
	
	# Start the snake at the center of the grid
	snake = [Vector2i(center_x, center_y)]
	grid[snake[0].x][snake[0].y] = SNAKE_BODY

func spawn_food():
	var x = randi() % GRID_SIZE.x
	var y = randi() % GRID_SIZE.y
	food_position = Vector2i(x, y)
	grid[food_position.x][food_position.y] = FOOD


func move_snake():
	if game_over:
		return

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
	# Mark the tail segments with -1 and the head with 1
	grid[new_head.x][new_head.y] = 1  # Mark the new head
	for i in range(1, snake.size()):
		var segment = snake[i]
		grid[segment.x][segment.y] = -1  # Mark the tail


func run_training_episode():
	initialize_grid()
	place_snake()
	spawn_food()
	game_over = false  # Ensure game_over is reset
	while not game_over:
		var state = get_state()
		var action = dqn.predict(state, 0, false)  # DQN decides the action
		apply_action(action)
		move_snake()

		if not game_over:
			var reward = 1 if snake[0] == food_position else 0
			var next_state = get_state()
			dqn.predict(next_state, reward, game_over)

func apply_action(action: int):
	var proposed_direction: Vector2i

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

	#print("Direction updated to:", direction)

func get_state() -> Array:
	var state: Array = []
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
		var action = dqn.predict(state, 0, false)
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
	print("Current Board:")
	for y in range(GRID_SIZE.y):
		var row = ""
		for x in range(GRID_SIZE.x):
			match grid[x][y]:
				EMPTY:
					row += ". "
				SNAKE_HEAD:
					if Vector2i(x, y) == snake[0]:
						row += "H "  # Mark the current head with 'H'
				SNAKE_BODY:
						row += "S "  # Mark the rest of the snake's body with 'S'
				FOOD:
					row += "F "
		print(row)
	print("\n")
