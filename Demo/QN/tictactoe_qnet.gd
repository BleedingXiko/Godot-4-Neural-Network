extends Node2D

var qt_x: DDQN
var qt_o: DDQN
var board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
var player = 1
var waiting_for_input = false
var current_action: int = -1

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var config = {
	"print_debug_info": true,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.00001,
	"min_exploration_probability": 0.15,
	"exploration_strategy": "softmax",  # Changed to epsilon_greedy
	"sampling_strategy": "sequential",
	"discounted_factor": 0.98,
	"decay_per_steps": 250,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 1000,
	"memory_capacity": 256,
	"batch_size": 25,
	"learning_rate": 0.0000001,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.01,
	"use_adam_optimizer": false,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-4,
	"early_stopping": false,
	"patience": 15,
	"save_path": "res://dqn_snake.data",
	"smoothing_window": 100,
	"check_frequency": 10,
	"minimum_epochs": 200,
	"improvement_threshold": 0.00005,
	"use_gradient_clipping": true,
	"gradient_clip_value": 0.3,
	"initialization_type": "he",
	"loss_function_type": "mse",
}

var x_wins: int = 0
var o_wins: int = 0
var draws: int = 0

@onready var input_timer: Timer = $Timer

func _ready() -> void:
	# Initialize agents for X and O
	qt_x = DDQN.new(config)
	qt_x.add_layer(9)
	qt_x.add_layer(28, ACTIVATIONS.RELU)
	qt_x.add_layer(9, ACTIVATIONS.LINEAR)

	qt_o = DDQN.new(config)
	qt_o.add_layer(9)
	qt_o.add_layer(19, ACTIVATIONS.RELU)
	qt_o.add_layer(9, ACTIVATIONS.LINEAR)
	
	#qt_x.load("user://ddqnet_x_ttt2.data", config)
	#qt_o.load("user://ddqnet_o_ttt2.data", config)

	train_networks()

	#$VisualizeNet.visualize(qt_x.neural_network)
	#$VisualizeNet2.visualize(qt_o.neural_network)
	qt_x.save("user://ddqnet_x_ttt2.data")
	qt_o.save("user://ddqnet_o_ttt2.data")

	qt_x.load("user://ddqnet_x_ttt2.data")
	qt_o.load("user://ddqnet_o_ttt2.data")

	print("Training complete. Ready to play!")
	play_against_ai()
	o_wins = 0
	x_wins = 0
	draws = 0

func train_game():
	var done = false
	var previous_reward = 0.0
	var current_reward: float

	while not done:
		var state = board.duplicate()

		# Choose action based on the current player
		var action: int
		if player == 1:
			action = qt_x.choose_action(state)
		else:
			action = qt_o.choose_action(state)

		if update_board(player, action):
			current_reward = determine_value_training(board, player)
			done = check_end_game()

			previous_reward = current_reward
			if player == 1:
				qt_x.train(board, previous_reward, done)
			else:
				qt_o.train(board, previous_reward, done)
		else:
			# Invalid move, punish player but don't end the game
			current_reward = -0.5
			if player == 1:
				qt_x.train(board, current_reward, false)
			else:
				qt_o.train(board, current_reward, false)
			continue

		if not done:
			player = switch_player(player)

	if player == 1:
		qt_x.train(board, previous_reward, true)
	else:
		qt_o.train(board, previous_reward, true)

	update_win_counts(has_winner(board))

func train_networks():
	for i in range(500):
		init_board()
		train_game()

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
			# Train DQN after player's move
			if not check_end_game():
				train_dqn_after_move()
		else:
			var action = qt_o.choose_action(board)
			if update_board(player, action):
				print_board()
				if check_end_game():
					await get_tree().create_timer(2.0).timeout
					continue  # Start a new game
				train_dqn_after_move()
				player = switch_player(player)

func train_dqn_after_move():
	var reward = determine_value_training(board, player)
	qt_x.train(board.duplicate(), reward, check_end_game())


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
	return 0.1  # Increased the reward for a draw to encourage more strategic play

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
