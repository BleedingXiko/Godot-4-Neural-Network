extends Node2D

var qnet: QNetwork
var grid_size: Vector2 = Vector2(10, 10)
var snake = []
var food
@onready var food_timer = $Timer
@onready var game_timer = $Timer2
@onready var score_label = $Label
var score = 0
var previous_reward: float = 0.0
var snake_direction = Vector2(0, -1)
var tile_size = 20

var manhattan_distance = 0

var ACTIVATIONS: Dictionary = {
	"SIGMOID": {
		"function": Callable(Activation, "sigmoid"),
		"derivative": Callable(Activation, "dsigmoid"),
		"name": "SIGMOID",
	},
	"RELU": {
		"function": Callable(Activation, "relu"),
		"derivative": Callable(Activation, "drelu"),
		"name": "RELU"
	},
	"TANH": {
		"function": Callable(Activation, "tanh_"),
		"derivative": Callable(Activation, "dtanh"),
		"name": "TANH"
	},
	"ARCTAN": {
		"function": Callable(Activation, "arcTan"),
		"derivative": Callable(Activation, "darcTan"),
		"name": "ARCTAN"
	},
	"PRELU": {
		"function": Callable(Activation, "prelu"),
		"derivative": Callable(Activation, "dprelu"),
		"name": "PRELU"
	},
	"ELU": {
		"function": Callable(Activation, "elu"),
		"derivative": Callable(Activation, "delu"),
		"name": "ELU"
	},
	"SOFTPLUS": {
		"function": Callable(Activation, "softplus"),
		"derivative": Callable(Activation, "dsoftplus"),
		"name": "SOFTPLUS"
	}
}

func _ready():
	qnet = QNetwork.new(5, [8], 4, ACTIVATIONS.TANH, ACTIVATIONS.SIGMOID, true, true) # 4 actions
	qnet.memory_capacity = 800
	qnet.batch_size = 128
	qnet.is_learning = true
	qnet.min_exploration_probability = 0.05
	qnet.learning_rate = 0.01
	qnet.decay_per_steps = 300
	#qnet.print_debug_info = true
	create_grid()
	reset_game()
	setup_timer()
	update_manhattan_distance()

func update_manhattan_distance():
	var distance_x = abs(snake[0].position.x - food.position.x)
	var distance_y = abs(snake[0].position.y - food.position.y)
	manhattan_distance = int(distance_x + distance_y)
func setup_timer():
	food_timer.wait_time = 50
	food_timer.autostart = true

func _on_food_timer_timeout():
	spawn_food()

func create_grid():
	for x in range(grid_size.x):
		for y in range(grid_size.y):
			var tile = Sprite2D.new()
			tile.texture = load("res://icon.svg") # Default Godot icon
			tile.modulate = Color(0.5, 0.5, 0.5) # Grey color for grid
			tile.position = Vector2(x, y) * tile_size
			tile.scale = Vector2(0.15, 0.15)
			add_child(tile)

func reset_game():
	for segment in snake:
		segment.queue_free()
	snake.clear()

	spawn_snake()
	spawn_food()
	score = 0
	update_score()

func spawn_snake():
	var snake_head = Sprite2D.new()
	snake_head.texture = load("res://icon.svg")
	snake_head.modulate = Color(1, 0, 0) # Red color for snake head
	snake_head.position = grid_size / 2 * tile_size
	snake_head.scale = Vector2(0.15, 0.15)
	add_child(snake_head)
	snake.append(snake_head)

func spawn_food():
	for child in get_children():
		if child.is_in_group("food"):
			child.queue_free()
	food = Sprite2D.new()
	food.add_to_group("food")
	food.texture = load("res://icon.svg")
	food.modulate = Color(0, 1, 0) # Green color for food
	var food_position = Vector2(randi_range(0, grid_size.x - 1), randi_range(0, grid_size.y - 1)) * tile_size
	food.position = food_position
	food.scale = Vector2(0.15, 0.15)
	add_child(food)


func _on_game_timeout():
	var action = qnet.predict(get_state(), previous_reward)
	previous_reward = get_reward()
	move_snake(action)
	update_score()

func get_state():
	var state = []
	var snake_pos = snake[0].position / tile_size
	var food_pos = food.position / tile_size
	state.append(snake_pos.x)
	state.append(snake_pos.y)
	state.append(food_pos.x)
	state.append(food_pos.y)
	state.append(score)
#	state.append(snake_direction.x) # Add snake's current direction to the state
#	state.append(snake_direction.y)
	return state

func get_reward():
	var reward = 0
	var new_distance_x = abs(snake[0].position.x - food.position.x)
	var new_distance_y = abs(snake[0].position.y - food.position.y)
	var new_manhattan_distance = int(new_distance_x + new_distance_y)

	if snake[0].position == food.position:
		score += 1
		print("ai won")
		food.queue_free()
		spawn_food()
		reward += 50
	elif snake[0].position.x < 0 or snake[0].position.x >= grid_size.x * tile_size or snake[0].position.y < 0 or snake[0].position.y >= grid_size.y * tile_size:
		reward += -20
	else:
		if new_manhattan_distance < manhattan_distance:
			reward += 5
		elif new_manhattan_distance > manhattan_distance:
			reward += -10

	manhattan_distance = new_manhattan_distance  # Update the global distance

	return reward


func move_snake(direction):
	match direction:
		0: snake_direction = Vector2(0, -1)
		1: snake_direction = Vector2(0, 1)
		2: snake_direction = Vector2(-1, 0)
		3: snake_direction = Vector2(1, 0)

	var new_position = snake[0].position + snake_direction * tile_size
	if is_position_valid(new_position):
		snake[0].position = new_position
	else:
		reset_game()

func is_position_valid(position: Vector2) -> bool:
	return position.x >= 0 and position.x < grid_size.x * tile_size and position.y >= 0 and position.y < grid_size.y * tile_size

	# Implement self-collision logic if the snake can grow

func update_score():
	score_label.text = "Score: " + str(score)

