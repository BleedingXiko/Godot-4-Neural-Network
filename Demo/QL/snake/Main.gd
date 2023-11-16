extends Node2D

var qnet: QNetwork
var grid_size: Vector2 = Vector2(6, 6)
var snake = []
var snake_body = []  # Holds the snake body parts
var food
@onready var food_timer = $Timer
@onready var game_timer = $Timer2
@onready var score_label = $Label
var score = 0
var previous_reward: float = 0.0
var snake_direction = Vector2(0, -1)
var tile_size = 20
var manhattan_distance = 0

var ACTIVATIONS = Activation.new().functions

var q_network_config = {
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.01,
	"min_exploration_probability": 0.05,
	"discounted_factor": 0.9,
	"learning_rate": 0.01,
	"decay_per_steps": 100,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 500,
	"memory_capacity": 300,
	"batch_size": 64,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": true,
}

func _ready():
	qnet = QNetwork.new(12, [8], 4, ACTIVATIONS.RELU, ACTIVATIONS.SIGMOID, q_network_config) # 4 actions
	create_grid()
	reset_game()
	setup_timer()
	update_manhattan_distance()

func _input(event):
	if event.is_action_pressed("ui_up"):
		qnet.save("./qnetwork.data")
	if event.is_action_pressed("ui_down"):
		qnet.load("./qnetwork.data", true)

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
	for body_part in snake_body:
		body_part.queue_free()
	snake_body.clear()

	spawn_snake()
	spawn_food()
	score = 0
	update_score()

func spawn_snake():
	var snake_head = Sprite2D.new()
	snake_head.texture = load("res://icon.svg")
	snake_head.modulate = Color(1, 0, 0) # Red color for snake head
	var snake_position = Vector2(randi_range(0, grid_size.x - 1), randi_range(0, grid_size.y - 1)) * tile_size
	snake_head.position = snake_position#
	snake_head.scale = Vector2(0.15, 0.15)
	add_child(snake_head)
	snake.append(snake_head)

	#var body_part = create_snake_body_part(snake_head.position - Vector2(0, tile_size))
	#snake_body.append(body_part)

func create_snake_body_part(position: Vector2) -> Sprite2D:
	var part = Sprite2D.new()
	part.texture = load("res://icon.svg")
	part.modulate = Color(0.5, 0, 0) # Yellow color for snake body
	part.position = position
	part.scale = Vector2(0.15, 0.15)
	add_child(part)  # Add the body part to the scene
	return part


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
	state.append(snake[0].position.x / (grid_size.x * tile_size - 1))
	state.append(snake[0].position.y / (grid_size.x * tile_size - 1))
	state.append(food.position.x / (grid_size.x * tile_size - 1))
	state.append(food.position.y / (grid_size.x * tile_size - 1))
	state.append(snake_direction.x / (grid_size.x * tile_size - 1))
	state.append(snake_direction.y / (grid_size.x * tile_size - 1))

	# Include positions of the three closest body parts
	for i in range(3):
		if i < snake_body.size():
			state.append(snake_body[i].position.x / (grid_size.x * tile_size - 1))
			state.append(snake_body[i].position.y / (grid_size.x * tile_size - 1))
		else:
			state.append(0)  # Placeholder for x-coordinate
			state.append(0)  # Placeholder for y-coordinate
	#print(state)
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
		grow_snake()
		reward += 10
	elif snake[0].position.x < 0 or snake[0].position.x >= grid_size.x * tile_size or snake[0].position.y < 0 or snake[0].position.y >= grid_size.y * tile_size:
		reward += -5
	else:
		if new_manhattan_distance < manhattan_distance:
			reward += 1
		elif new_manhattan_distance > manhattan_distance:
			reward += -0.5

	# Check for self-collision
	for body_part in snake_body:
		if snake[0].position == body_part.position:
			reward += -10
			reset_game()
			break

	manhattan_distance = new_manhattan_distance
	return reward

func grow_snake():
	var last_body_part
	if snake_body.size() == 0:
		last_body_part = snake[0]
	else:
		last_body_part = snake_body[-1]
	var new_body_part_position = last_body_part.position - snake_direction * tile_size
	var new_body_part = create_snake_body_part(new_body_part_position)
	snake_body.append(new_body_part)


func _get_position(body_part):
	return body_part.position


func move_snake(direction):
	var new_direction = Vector2.ZERO
	match direction:
		0: new_direction = Vector2(0, -1)
		1: new_direction = Vector2(0, 1)
		2: new_direction = Vector2(-1, 0)
		3: new_direction = Vector2(1, 0)

	# Prevent reversing direction
	if new_direction + snake_direction != Vector2.ZERO:
		snake_direction = new_direction

	var new_head_position = snake[0].position + snake_direction * tile_size

	if is_position_valid(new_head_position):
		# Move body parts
		for i in range(snake_body.size() - 1, 0, -1):
			snake_body[i].position = snake_body[i - 1].position
		if snake_body.size() > 0:
			snake_body[0].position = snake[0].position

		# Move head
		snake[0].position = new_head_position
	else:
		reset_game()


func is_position_valid(position: Vector2) -> bool:
	return position.x >= 0 and position.x < grid_size.x * tile_size and position.y >= 0 and position.y < grid_size.y * tile_size

func update_score():
	score_label.text = "Score: " + str(score)

