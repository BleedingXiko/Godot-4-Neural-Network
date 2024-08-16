extends Node2D

var qt_x: QTable
var qt_o: QTable
var board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
var player = 1

var x_wins: int = 0
var o_wins: int = 0
var draws: int = 0

var waiting_for_input = false
var current_action: int = -1

@onready var input_timer: Timer = $Timer

var q_table_config = {
	"print_debug_info": false,
	"is_learning": true,
	"action_threshold": 0.25,
	"exploration_decreasing_decay": 0.0005,
	"exploration_strategy": "ucb",
	"exploration_parameter": 0.3,
	"min_exploration_probability": 0.15,
	"discounted_factor": 0.9,
	"learning_rate": 0.1,
	"decay_per_steps": 100,
	"max_state_value": 2,
	"random_weights": false,
}

func _ready() -> void:
	qt_x = QTable.new()
	qt_x.init(9 * 3, 9, q_table_config)
	qt_o = QTable.new()
	qt_o.init(9 * 3, 9, q_table_config)
	
	#for i in range(20000):
		#init_board()
		#play_game()
	# Load previously saved QTables for gameplay
	qt_o.load('user://qt_o.data', {"is_learning": false,"exploration_strategy": "ucb",})

	print("Ready to play against AI!")
	x_wins = 0
	o_wins = 0
	draws = 0
	play_against_ai()

func hash_state(state_array: Array, hash_range: int) -> int:
	var hash_value := 0
	var prime := 31

	for state in state_array:
		hash_value = (hash_value * prime + state) % hash_range

	return hash_value

func play_game():
	var done = false
	var player_turn = randi_range(1, 2)
	var previous_reward_x = -100.0
	var previous_reward_o = -100.0
	var current_reward: float
	
	while not done:
		var state = hash_state(board, 9 * 3)
		var action: int
		
		if player_turn == 1:
			action = qt_x.predict([state], previous_reward_x)
		else:
			action = qt_o.predict([state], previous_reward_o)

		if update_board(player_turn, action):
			current_reward = determine_value_training(board, player_turn)

			if player_turn == 1:
				# Update X's rewards
				previous_reward_x = (previous_reward_x + current_reward) / 2.0
			else:
				# Update O's rewards
				previous_reward_o = (previous_reward_o + current_reward) / 2.0

			done = current_reward != 0.0  # The game ends if there's a win/loss or draw

			if not done:
				player_turn = switch_player(player_turn)
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
		current_action = -1
		
func process_player_move(action: int):
	if action >= 0 and action < 9 and update_board(player, action):
		print_board()
		check_end_game()
		player = switch_player(player)  # Switch to AI
	else:
		print("Invalid move. Try again.")

func determine_value_training(_board: Array, player_turn: int) -> float:
	var result = has_winner(_board)
	
	if result == 1:  # X wins
		return 1.0 if player_turn == 1 else -1.0  # Positive reward for X, negative for O
	elif result == 2:  # O wins
		return 1.0 if player_turn == 2 else -1.0  # Positive reward for O, negative for X
	return 0.0  # Draw

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
			var action = qt_o.predict([hash_state(board, 9 * 3)], 0)
			if update_board(player, action):
				print_board()
				if check_end_game():
					await get_tree().create_timer(2.0).timeout
					continue  # Start a new game
				player = switch_player(player)
			else:
				print("AI made an invalid move, retrying...")
				if randf() > 0.5:
						if update_board(player, randi() % 9):
							print_board()
							if check_end_game():
								await get_tree().create_timer(2.0).timeout
								continue  # Start a new game
							player = switch_player(player)
				# AI retries until a valid move is made

func init_board():
	for i in range(9):
		board[i] = 0


func update_board(player: int, index: int) -> bool:
	if index >= 0 and index < 9 and board[index] == 0:
		board[index] = player
		return true
	return false

func check_end_game() -> bool:
	var winner = has_winner(board)
	if winner != 0:
		update_win_counts(winner)
		init_board()  # Reset the board for a new game
		return true
	return false

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

func update_win_counts(winner: int):
	if winner == 1:
		x_wins += 1
	elif winner == 2:
		o_wins += 1
	else:
		draws += 1
	print("X Wins: ", x_wins, " O Wins: ", o_wins, " Draws: ", draws)
	print_board()

func switch_player(current_player: int) -> int:
	return 2 if current_player == 1 else 1

func print_board():
	var spacing = "\n\n"
	var row = ""
	for i in range(9):
		var symbol = "_"
		if board[i] == 1:
			symbol = "X"
		elif board[i] == 2:
			symbol = "O"
		row += symbol + " "
		if (i + 1) % 3 == 0:
			print(row.strip_edges())
			row = ""
	print(spacing)
