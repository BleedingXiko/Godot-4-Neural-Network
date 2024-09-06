extends Node2D

var nnas: NeuralNetworkAdvanced

@onready var input: ColorRect = $input1
@onready var output: ColorRect = $output

var config = {
	"learning_rate": 0.01,  # Small learning rate for stable training
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.005,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-4,
	"early_stopping": false,
	"patience": 4000,  # Increase patience for more complex training cycles
	"minimum_epochs": 2000,  # Longer training for better convergence
	"smoothing_window": 500,  # Smooth over more epochs for stability
	"check_frequency": 10,  # Less frequent checks for performance
	"save_path": "res://nn.best.data",
	
	# Gradient Clipping
	"use_gradient_clipping": true,
	"gradient_clip_value": 2.0,
	"loss_function_type": "mse",  # Regression task: use Mean Squared Error

	# Weight Initialization
	"initialization_type": "xavier"  # Using 'he' initialization for ReLU-based activations
}

var neural_network = NeuralNetworkAdvanced.new(config)

var activation_functions = [
	"SIGMOID",
	"RELU",
	"TANH",
	"SWISH",
	"MISH"
]

func _ready() -> void:
	randomize()
	seed(randi())
	nnas = NeuralNetworkAdvanced.new(config)
	
	# Input layer with 3 neurons (for RGB values)
	nnas.add_layer(3)

	# Single hidden layer with 8 neurons using RELU
	nnas.add_layer(6, nnas.ACTIVATIONS.SIGMOID)

	# Output layer with 3 neurons (for RGB values), using linear activation
	nnas.add_layer(3, nnas.ACTIVATIONS.LINEAR)

	# Visualize the network structure
	$VisualizeNet.visualize(nnas)
	$VisualizeNet2.visualize(nnas)

func _input(event: InputEvent) -> void:
	if event.is_action("predict"):
		start_training()

func start_training() -> void:
	var random_color: Array = [randi() % 256, randi() % 256, randi() % 256]
	var complementary_color: Array = [255 - random_color[0], 255 - random_color[1], 255 - random_color[2]]

	# Normalize the input and output to the range [0, 1]
	var normalized_input = random_color.map(func(x): return x / 255.0)
	var normalized_output = complementary_color.map(func(x): return x / 255.0)

	# Update the ColorRect objects to visualize the input
	input.color = Color(normalized_input[0], normalized_input[1], normalized_input[2])

	# Predict using the neural network (output is still normalized)
	var prediction = nnas.predict(normalized_input)

	# Denormalize the predicted output to the range [0, 255] for RGB values
	var denormalized_prediction = prediction.map(func(x): return clamp(x * 255, 0, 255))

	# Update the output ColorRect using the denormalized RGB values
	output.color = Color(denormalized_prediction[0] / 255.0, denormalized_prediction[1] / 255.0, denormalized_prediction[2] / 255.0)

	# Train the network
	for i in range(1):
		nnas.train(normalized_input, normalized_output)

	# After training, visualize the network
	$VisualizeNet2.visualize(nnas)
