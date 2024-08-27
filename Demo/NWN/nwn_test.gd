extends Node2D

func _ready() -> void:
	var master_config = {
		"learning_rate": 0.001,
		"l2_regularization_strength": 0.001,
		"use_l2_regularization": false,
	}

	var nnwn = NeuralNetworkInNetwork.new(master_config)

	# Add sub-network layers with hidden layers
# Adjusted to ensure the output sizes match the expected input sizes
	nnwn.add_nin_layer(3, 2, [0])
	nnwn.add_nin_layer(4, 3, [0])  # Sub-networks have hidden layers with 4 and 3 neurons
	   # This layer takes the output of 3 neurons from the previous layer

	# Add master network layers with hidden layers
	nnwn.add_master_network_layer(1, [6, 3], nnwn.ACTIVATIONS.SIGMOID)  # Master network with two hidden layers (6 and 3 neurons)

	# Example input
	for i in range(20000):
		var input_data = [0.5, 0.2]
		var prediction = nnwn.predict(input_data)
		print("Prediction: ", prediction)

		# Training example
		var target_data = [1.0]
		nnwn.train(input_data, target_data)
