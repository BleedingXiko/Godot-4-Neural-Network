extends Node2D
var training_inputs = []
var training_outputs = []
var num_samples = 200


# Initialize the MLP
var config = {
	"learning_rate": 0.000001,  # Lower the learning rate
	"use_l2_regularization": true,  # Enable L2 regularization
	"l2_regularization_strength": 0.0001,  # Small regularization
	"use_adam_optimizer": true,  # Adam optimizer
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-8,
	"early_stopping": false,
	"initialization_type": "xavier",  # Use Xavier initialization
	"loss_function_type": "mse",  # MSE for regression
	"use_gradient_clipping": true,
	"gradient_clip_value": 0.4,
	}
var mlp = NeuralNetworkAdvanced.new(config)

func _ready():
	print("Training started...")  # Debug print to see if _ready is triggered
	# Generate training data for sin(x)
	for i in range(num_samples):
		var x = randf_range(-PI, PI)
		training_inputs.append([x])
		training_outputs.append([sin(x)])

	mlp.add_layer(1)  # Input layer with 1 neuron
	mlp.add_layer(10, mlp.ACTIVATIONS.TANH)  # Hidden layer
	mlp.add_layer(1, mlp.ACTIVATIONS.LINEAR)  # Output layer

	print("Training MLP...")
	for epoch in range(10000):
		for i in range(training_inputs.size()):
			var input = training_inputs[i]
			var target = training_outputs[i]
			mlp.train(input, target)
			var p = mlp.predict(input)
			print("Input: ", input, " Prediction: ", p, " Actual: ", sin(input[0]))  # Ensure this prints


	print("Testing and updating label...")
	_test_and_update_label(mlp)

func _test_and_update_label(mlp):
	for i in range(-PI, PI, 0.1):
		var input = [i]
		var output = mlp.predict(input)
		print("Input: ", i, " Prediction: ", output, " Actual: ", sin(i))  # Ensure this prints
		$Label.text = "Input: " + str(i) + " Prediction: " + str(output) + " Actual: " + str(sin(i))

		await get_tree().create_timer(0.5).timeout  # Add delay for label update
