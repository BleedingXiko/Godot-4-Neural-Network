extends Node2D

var q_network: DDQN
var current_state: Array = [0, 0, 0]  # Start with a 3-bit binary number [000]
var target_state: Array = [1, 1, 1]  # The target is to reach [111]

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var q_network_config = {
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.01,
	"min_exploration_probability": 0.1,
	"exploration_strategy": "epsilon_greedy",
	"sampling_strategy": "sequential",
	"discounted_factor": 0.9,
	"decay_per_steps": 100,
	"use_replay": false,
	"is_learning": true,
	"use_target_network": false,
	"memory_capacity": 1024,
	"batch_size": 16,
	"learning_rate": 0.05,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7,
	"early_stopping": false,  # Enable or disable early stopping
	"patience": 500,          # Number of epochs with no improvement after which training will be stopped
	"save_path": "res://earlystoptestbinflip.data",  # Path to save the best model
	"smoothing_window": 10,  # Number of epochs to average for loss smoothing
	"check_frequency": 50,    # Frequency of checking early stopping condition
	"minimum_epochs": 1000,   # Minimum epochs before early stopping can trigger
	"improvement_threshold": 0.005  # Minimum relative improvement required to reset patience
}

func _ready():
	# Initialize the QNetwork
	seed(26)
	randomize()
	q_network = DDQN.new(q_network_config)
	q_network.add_layer(3)  # Input layer: 3 bits
	q_network.add_layer(5, ACTIVATIONS.RELU)  # Hidden layer
	q_network.add_layer(3, ACTIVATIONS.SIGMOID)  # Output layer: 3 possible actions (flip each bit)
	#q_network.load(q_network.neural_network.save_path)
	$VisualizeNet.visualize(q_network.neural_network)
	for i in range(8):
		train_network()
	play_binary_counter()

func train_network():
	var done = false
	var player_turn = randi_range(1, 2)
	var previous_reward = -100.0
	var current_reward: float

	reset_state()
	done = false
	previous_reward = 0.0
	
	while not done:
		var action = q_network.choose_action(current_state)
		var new_state = current_state.duplicate()
		
		# Perform the action (flip the chosen bit)
		new_state[action] = 1 - new_state[action]
		
		# Calculate the reward
		var reward = calculate_reward(new_state)
		
		# Check if the goal is reached
		if new_state == target_state:
			done = true
			
		# Update the Q-network with the new experience
		q_network.train(new_state, reward, done)
		
		# Update the current state
		current_state = new_state
		
		$VisualizeNet2.visualize(q_network.neural_network)

func calculate_reward(state: Array) -> float:
	# Reward is based on how close the current state is to the target state
	if state == target_state:
		return 1.0  # Goal reached
	elif state == current_state:
		return -0.1  # No change (unlikely but possible)
	else:
		return 0.1  # Closer to the goal

func play_binary_counter():
	reset_state()
	print("Starting to play...")
	
	while current_state != target_state:
		print("Current state: %s" % str(current_state))
		var action = q_network.choose_action(current_state)
		current_state[action] = 1 - current_state[action]
		print("Action taken: Flip bit %d" % action)
		print("New state: %s" % str(current_state))
		
	print("Goal reached! Final state: %s" % str(current_state))

func reset_state():
	# Start from the initial state [000]
	current_state = [0, 0, 0]
