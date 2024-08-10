extends Node2D

var qt_x: QNetwork
var qt_o: QNetwork
var board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
var player = 1

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var q_network_config = {
	"print_debug_info": false,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.01,
	"min_exploration_probability": 0.05,
	"discounted_factor": 0.95,
	"decay_per_steps": 250,
	"use_replay": false,
	"is_learning": true,
	"use_target_network": false,
	"update_target_every_steps": 1500,
	"memory_capacity": 800,
	"batch_size": 256,
	"learning_rate": 0.00000001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

var x_wins: int = 0
var o_wins: int = 0
var draws: int = 0

func _ready() -> void:
	qt_x = QNetwork.new(q_network_config)
	qt_x.add_layer(9) #input nodes
	qt_x.add_layer(12, ACTIVATIONS.RELU) #hidden layer
	qt_x.add_layer(8, ACTIVATIONS.TANH) #hidden layer
	qt_x.add_layer(9, ACTIVATIONS.SIGMOID) # 4 actions
	
	qt_o = QNetwork.new(q_network_config)
	qt_o.add_layer(9) #input nodes
	qt_o.add_layer(6, ACTIVATIONS.RELU) #hidden layer
	qt_o.add_layer(12, ACTIVATIONS.TANH) #hidden layer
	qt_o.add_layer(9, ACTIVATIONS.SIGMOID) # 4 actions

	for i in range(50000):  # Learning phase
		init_board()
		play_game()

	qt_x.save('user://qnet_x.data')
	qt_o.save('user://qnet_o.data')

func init_board():
	for i in range(9):
		board[i] = 0

func hash_state(state_array: Array, hash_range: int) -> int:
	var hash_value := 0
	var prime := 31

	for state in state_array:
		hash_value = (hash_value * prime + state) % hash_range

	return hash_value

func play_game():
	var done = false
	var player_turn = randi_range(1, 2)
	var previous_reward = -100
	
	while not done:
		var state = board
		var action: int

		if player_turn == 1:
			action = qt_x.predict(state, previous_reward)
		else:
			action = qt_o.predict(state, previous_reward)

		if update_board(player_turn, action):
			var reward = determine_value(board)

			# Update the previous reward based on the outcome
			previous_reward = reward
			done = reward != 0.5  # The game ends if there's a win/loss or draw

			if not done:
				player_turn = switch_player(player_turn)
		else:
			# Invalid move, punish player
			previous_reward = -0.75
			done = true

	update_win_counts(has_winner(board))

func update_board(player: int, index: int) -> bool:
	if board[index] == 0:
		board[index] = player
		return true
	return false

func determine_value(_board: Array) -> float:
	var result = has_winner(_board)
	if result == 1:
		return 1.0  # Player 1 wins
	elif result == 2:
		return 0.0  # Player 2 wins
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
	var spacing = "\n\n"
	var row = ""
	for i in range(9):
		var symbol = "_"
		if board[i] == 1:
			symbol = "X"
		elif board[i] == 2:
			symbol = "O"

		# Append the symbol to the row with a space
		row += symbol + " "

		# Check if it's the end of the row
		if (i + 1) % 3 == 0:
			print(row.strip_edges())
			row = ""

	# Print additional spacing after the board
	print(spacing)
		
	# Print the statistics
	print("Player X Wins: " + str(x_wins))
	print("Player O Wins: " + str(o_wins))
	print("Draws: " + str(draws))
	print(spacing)

func update_win_counts(winner: int) -> void:
	if winner == 1:
		print_board()
		x_wins += 1
	elif winner == 2:
		print_board()
		o_wins += 1
	elif winner == -1:
		print_board()
		draws += 1
	

