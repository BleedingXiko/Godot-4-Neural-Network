extends Node2D

var qnet: QNetwork
var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var row: int = 0
var column: int = 0

@onready var six = $bck/six
@onready var five = $bck/five
@onready var tf = $bck/tf

var reward_states = [5, 6, 34]
var target: int = reward_states.pick_random()
var punish_states = [3, 18, 21, 28, 31]

var current_state: Array = []
var previous_reward: float = 0.0


var done: bool = false

var q_network_config = {
	"print_debug_info": true,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.005,
	"min_exploration_probability": 0.15,
	"discounted_factor": 0.95,
	"decay_per_steps": 250,
	"use_replay": false,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 1500,
	"memory_capacity": 2048,
	"batch_size": 512,
	"learning_rate": 0.0001, 
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

var initial_grid := [
	0, 2, 0, 0, 0, 0,
	0, 0, 0, 0, 2, 0,
	2, 0, 0, 2, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0,
	0, 0, 0, 2, 0, 0
]

var grid := []

func _ready() -> void:
	qnet = QNetwork.new(q_network_config)
	qnet.add_layer(36)
	qnet.add_layer(8, ACTIVATIONS.SWISH)
	qnet.add_layer(10, ACTIVATIONS.RELU)
	qnet.add_layer(4, ACTIVATIONS.SIGMOID)
	update_grid(row, column, target)

func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$Timer.wait_time = 0.5
	elif Input.is_action_just_pressed("ui_down"):
		$Timer.wait_time = 0.001

func hash_state(state_array: Array, hash_range: int) -> int:
	var hash_value := 0
	var prime := 31

	for state in state_array:
		hash_value = (hash_value * prime + state) % hash_range

	return hash_value

func print_grid():
	for i in range(6):
		var row_str := ""
		for j in range(6):
			row_str += str(grid[i * 6 + j]) + " "
		print(row_str)
	print("\n\n")

func _on_timer_timeout():
	#print_grid()
	current_state = grid
	var action_to_do: int = qnet.predict(current_state, previous_reward)
	if done:
		reset()
	
	previous_reward = 0.0
		
	if is_out_bound(action_to_do):
		previous_reward -= 0.75
		done = true
	elif row * 6 + column in punish_states:
		previous_reward -= 0.5
		done = true
	elif (row * 6 + column) == target:
		previous_reward += 1.0
		done = false
		target = reward_states.pick_random()
	else:
		previous_reward -= 0.05
		
	update_grid(row, column, target)
	$player.position = Vector2(96 * column + 16, 96 * (5 - row) + 16)  # Adjust for correct visual alignment
	$lr.text = str(qnet.exploration_probability)
	$target.text = str(target)

func update_grid(row: int, column: int, target: int):
	# Reset the grid to the initial configuration
	grid = initial_grid.duplicate()  # Make a copy of the initial grid

	# Set the player's new position
	grid[row * 6 + column] = 1

	# Convert target index to row and column
	var target_row = int(target / 6)
	var target_column = target % 6
	
	# Place the new target
	grid[target_row * 6 + target_column] = 3


func is_out_bound(action: int) -> bool:
	var _column := column
	var _row := row
	match action:
		0: _column -= 1  # Left
		1: _row += 1    # Down
		2: _column += 1  # Right
		3: _row -= 1    # Up
	if _column < 0 or _row < 0 or _column > 5 or _row > 5:
		return true
	else:
		column = _column
		row = _row
		return false

func reset():
	target = reward_states.pick_random()
	row = randi_range(0, 5)
	column = randi_range(0, 5)
	done = false
	$player.position = Vector2(96 * column + 16, 96 * (5 - row) + 16)  # Adjust for correct visual alignment
	update_grid(row, column, target)
	if target == 5:
		five.show()
		six.hide()
		tf.hide()
	elif target == 6:
		five.hide()
		six.show()
		tf.hide()
	elif target == 34:
		five.hide()
		six.hide()
		tf.show()
	


func _on_save_pressed():
	qnet.save('user://qnet.data')

func _on_load_pressed():
	qnet.load('user://qnet.data', false, 0.05)
