extends Node2D

var mlp: NeuralNetworkAdvanced

func _ready():
	# XOR Training Data
	var training_inputs = [
		[0.0, 0.0],
		[0.0, 1.0],
		[1.0, 0.0],
		[1.0, 1.0]
	]
	
	var training_outputs = [
		[0.0],
		[1.0],
		[1.0],
		[0.0]
	]

	# Initialize the MLP
	var config = {
	"learning_rate": 0.15,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.001,
	"use_adam_optimizer": false,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-8,
	"early_stopping": false,  # Enable or disable early stopping
	"patience": 100,          # Number of epochs with no improvement after which training will be stopped
	"save_path": "res://earlystoptest.data",  # Path to save the best model
	"smoothing_window": 100,  # Number of epochs to average for loss smoothing
	"check_frequency": 50,    # Frequency of checking early stopping condition
	"minimum_epochs": 1000,   # Minimum epochs before early stopping can trigger
	"improvement_threshold": 0.00005,  # Minimum relative improvement required to reset patience
	# Gradient Clipping
	"use_gradient_clipping": false,
	"gradient_clip_value": 1.0,
	"loss_function_type": "binary_cross_entropy",

	# Weight Initialization
	"initialization_type": "random",  # Options are "xavier" or "he"
}
	
	mlp = NeuralNetworkAdvanced.new(config)
	mlp.add_layer(2)  # Input layer with 2 neurons
	mlp.add_layer(4, mlp.ACTIVATIONS.SIGMOID)  # Hidden layer with 3 neurons
	mlp.add_layer(1, mlp.ACTIVATIONS.SIGMOID)  # Output layer with 1 neuron
	
	$VisualizeNet.visualize(mlp)
	# Train the network
	var epochs = 10000
	for epoch in range(epochs):
		var total_loss = 0.0
		for i in range(training_inputs.size()):
			var input = training_inputs[i]
			var target = training_outputs[i]
			mlp.train(input, target)
	
	# Test the trained network
	for i in range(training_inputs.size()):
		var input = training_inputs[i]
		var output = mlp.predict(input)
		print("Input: ", input, " Prediction: ", output)
	$VisualizeNet2.visualize(mlp)
