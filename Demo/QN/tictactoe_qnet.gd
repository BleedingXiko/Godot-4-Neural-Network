extends Node2D

var qt_x: QNetwork
var board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
var player = 1
var waiting_for_input = false
var current_action: int = -1

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var q_network_config = {
	"print_debug_info": false,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.005,
	"min_exploration_probability": 0.05,
	"exploration_strategy": "softmax",
	"discounted_factor": 1,
	"decay_per_steps": 250,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 3000,
	"memory_capacity": 2048,
	"batch_size": 128,
	"learning_rate": 0.000001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

var x_wins: int = 0
var o_wins: int = 0
var draws: int = 0

@onready var input_timer: Timer = $Timer

func _ready() -> void:
	qt_x = QNetwork.new(q_network_config)
	qt_x.add_layer(9)  # Input layer (implicitly)
	qt_x.add_layer(16, ACTIVATIONS.RELU)  # Hidden layer with ELU activation
	qt_x.add_layer(12, ACTIVATIONS.RELU)
	qt_x.add_layer(8, ACTIVATIONS.SWISH)   # Another hidden layer with ELU activation
	qt_x.add_layer(9, ACTIVATIONS.LINEAR)  # Output layer with no activation function (linear)
	#qt_x.load("user://qnet_ttt.data", q_network_config)
	train_networks()
	qt_x.save("user://qnet_ttt.data")
	qt_x.load("user://qnet_ttt.data")

	print("Training complete. Ready to play!")
	play_against_ai()
	o_wins = 0
	x_wins = 0
	draws = 0

func train_networks():
	for i in range(50000):
		init_board()
		train_game()

func train_game():
	var done = false
	var previous_reward = 0.0
	var current_reward: float

	while not done:
		var state = board.duplicate()

		# Predict action based on the current state and player
		var action: int = qt_x.predict(state, previous_reward, done)

		if update_board(player, action):
			current_reward = determine_value_training(board, player)
			done = check_end_game()  # End the game if there's a win/loss/draw

			# Update previous reward for the current player
			previous_reward = current_reward
		else:
			# Invalid move, punish player
			current_reward = -0.75
			done = true
			previous_reward = current_reward

		if not done:
			player = switch_player(player)

	# Ensure final state is processed
	qt_x.predict(board, previous_reward, true)
	update_win_counts(has_winner(board))

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("1"):
		current_action = 0
	elif event.is_action_pressed("2"):
		current_action = 1
	elif event.is_action_pressed("3"):
		current_action = 2
	elif event.is_action_pressed("4"):
		current_action = 3
	elif event.is_action_pressed("5"):
		current_action = 4
	elif event.is_action_pressed("6"):
		current_action = 5
	elif event.is_action_pressed("7"):
		current_action = 6
	elif event.is_action_pressed("8"):
		current_action = 7
	elif event.is_action_pressed("9"):
		current_action = 8

	if waiting_for_input and current_action != -1:
		waiting_for_input = false
		process_player_move(current_action)
		$Timer.stop()
		$Timer.emit_signal("timeout")


func play_against_ai():
	init_board()
	print_board()
	player = 1  # You start first as X

	while true:
		if player == 1:
			waiting_for_input = true
			input_timer.start()
			await input_timer.timeout
		else:
			var action = qt_x.predict(board, 0, false)
			if update_board(player, action):
				print_board()
				if check_end_game():
					await get_tree().create_timer(2.0).timeout
					continue  # Start a new game
				player = switch_player(player)
			else:
				if update_board(player, randi() % 9):
					print_board()
					if check_end_game():
						await get_tree().create_timer(2.0).timeout
						continue  # Start a new game
					player = switch_player(player)

func ai_move(ai_player_turn: int) -> int:
	var valid_move = false
	while not valid_move:
		var action = qt_x.predict(board, 0, false)
		if update_board(ai_player_turn, action):
			print_board()
			valid_move = true
	return ai_player_turn  # Return the current player as AI’s move is complete

func init_board():
	for i in range(9):
		board[i] = 0

func check_end_game() -> bool:
	var winner = has_winner(board)
	if winner != 0:
		update_win_counts(winner)
		init_board()  # Reset the board for a new game
		return true
	return false


func update_board(player: int, index: int) -> bool:
	if board[index] == 0:
		board[index] = player
		return true
	return false

func process_player_move(action: int):
	if update_board(player, action):
		print_board()
		player = switch_player(player)  # Switch to AI
	else:
		waiting_for_input = true  # Continue waiting for a valid input

func determine_value_training(_board: Array, player_turn: int) -> float:
	var result = has_winner(_board)

	if result == 1:  # X wins
		return 1.0 if player_turn == 1 else -1.0  # Positive reward for X, negative for O
	elif result == 2:  # O wins
		return 1.0 if player_turn == 2 else -1.0  # Positive reward for O, negative for X
	return 0.5  # Draw

func switch_player(player: int) -> int:
	return 2 if player == 1 else 1

func has_winner(_board: Array) -> int:
	for player in range(1, 3):
		# Check horizontal
		for i in range(3):
			if _board[i * 3] == player and _board[i * 3 + 1] == player and _board[i * 3 + 2] == player:
				return player

		# Check vertical
		for i in range(3):
			if _board[i] == player and _board[i + 3] == player and _board[i + 6] == player:
				return player

		# Check diagonals
		if (_board[0] == player and _board[4] == player and _board[8] == player) or (_board[2] == player and _board[4] == player and _board[6] == player):
			return player

	# Check for draw
	for i in range(9):
		if _board[i] == 0:
			return 0  # Game continues

	return -1  # Draw

func print_board():
	var spacing = "\n"
	var row = ""
	for i in range(9):
		var symbol = "_"
		if board[i] == 1:
			symbol = "X"
		elif board[i] == 2:
			symbol = "O"

		row += symbol + " "
		if (i + 1) % 3 == 0:
			print(row)
			row = ""
	print(spacing)

func update_win_counts(winner: int) -> void:
	if winner == 1:
		x_wins += 1
	elif winner == 2:
		o_wins += 1
	else:
		draws += 1

	print("X Wins: %d, O Wins: %d, Draws: %d" % [x_wins, o_wins, draws])
