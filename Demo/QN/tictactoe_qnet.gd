extends Node2D

var qt_x: QNetwork
var qt_o: QNetwork
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
	"discounted_factor": 0.95,
	"decay_per_steps": 250,
	"use_replay": false,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 1500,
	"memory_capacity": 800,
	"batch_size": 256,
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
	qt_x.add_layer(9)
	qt_x.add_layer(8, ACTIVATIONS.TANH)
	qt_x.add_layer(9, ACTIVATIONS.SIGMOID)
	
	qt_o = QNetwork.new(q_network_config)
	qt_o.add_layer(9)
	qt_o.add_layer(12, ACTIVATIONS.TANH)
	qt_o.add_layer(9, ACTIVATIONS.SIGMOID)

	# Uncomment these lines if you need to train and save the networks
	qt_o.load('user://qnet_o.data', false)
	#qt_x.load('user://qnet_x.data', true, 1.0)
	#train_networks()
	#qt_o.save('user://qnet_o.data')
	#qt_x.save('user://qnet_x.data')

	# After training, you can play against the AI
	print("Training complete. Ready to play!")
	play_against_ai()
	o_wins = 0
	x_wins = 0
	draws = 0

func train_networks():
	while qt_x.exploration_probability > qt_x.min_exploration_probability:
		init_board()
		train_game()

func train_game():
	var done = false
	var player_turn = randi_range(1, 2)
	var previous_reward_x = -100.0
	var previous_reward_o = -100.0
	var current_reward: float

	while not done:
		var state = board
		var action: int

		if player_turn == 1:
			action = qt_x.predict(state, previous_reward_x)
		else:
			action = qt_o.predict(state, previous_reward_o)

		if update_board(player_turn, action):
			current_reward = determine_value_training(board, player_turn)

			if player_turn == 1:
				previous_reward_x = (previous_reward_x + current_reward) / 2.0
			else:
				previous_reward_o = (previous_reward_o + current_reward) / 2.0

			done = current_reward != 0.5  # The game ends if there's a win/loss or draw
		else:
			# Invalid move, punish player
			current_reward = -0.75

			if player_turn == 1:
				previous_reward_x = (previous_reward_x + current_reward) / 2.0
			else:
				previous_reward_o = (previous_reward_o + current_reward) / 2.0

			done = true

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

func play_against_ai():
	while true:
		init_board()
		print_board()
		var done = false
		player = 1  # Reset player to X at the start of the game

		while not done:
			if player == 1:
				waiting_for_input = true
				input_timer.start()
				await input_timer.timeout
				if current_action != -1:
					var action = current_action
					current_action = -1
					if update_board(player, action):
						print_board()
						var winner = determine_value(board)
						if winner != 0:  # Game is over (either a win or a draw)
							done = true
							update_win_counts(winner)
							await get_tree().create_timer(2.0).timeout
							break
						player = switch_player(player)
					else:
						print("Invalid move! Try again.")
						waiting_for_input = true
						continue
				else:
					continue
			else:
				print("AI's turn.")
				player = await ai_move(player)
				var winner = determine_value(board)
				if winner != 0:  # Game is over (either a win or a draw)
					done = true
					update_win_counts(winner)
					await get_tree().create_timer(2.0).timeout
					break
				player = switch_player(player)

		print("Starting a new game...")

func ai_move(ai_player_turn: int) -> int:
	var valid_move = false
	while not valid_move:
		var action = qt_o.predict(board, 0)
		if update_board(ai_player_turn, action):
			print_board()
			valid_move = true
		else:
			print("AI made an invalid move, retrying.")
	
	return ai_player_turn  # Return the current player as AI’s move is complete

func init_board():
	for i in range(9):
		board[i] = 0

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
		print("Invalid move. Try again.")
		waiting_for_input = true  # Continue waiting for a valid input

func determine_value(_board: Array) -> int:
	var result = has_winner(_board)
	if result == 1:  # X wins
		return 1
	elif result == 2:  # O wins
		return 2
	elif result == -1:  # Draw
		return -1  # Draw
	return 0  # Game continues

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
	var lines = [
		[0, 1, 2], [3, 4, 5], [6, 7, 8],
		[0, 3, 6], [1, 4, 7], [2, 5, 8],
		[0, 4, 8], [2, 4, 6]
	]
	for line in lines:
		if _board[line[0]] != 0 and _board[line[0]] == _board[line[1]] and _board[line[1]] == _board[line[2]]:
			return _board[line[0]]
	for i in _board:
		if i == 0:
			return 0  # Continue playing
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
