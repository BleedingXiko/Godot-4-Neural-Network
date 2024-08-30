extends Node2D

var nnas: NeuralNetworkAdvanced

@onready var input: ColorRect = $input1
@onready var output: ColorRect = $output

var config = {
	"learning_rate": 0.1,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.001,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7,
	"early_stopping": true,  # Enable or disable early stopping
	"patience": 500,          # Number of epochs with no improvement after which training will be stopped
	"save_path": "res://earlystoptestcolo.data",  # Path to save the best model
	"smoothing_window": 10,  # Number of epochs to average for loss smoothing
	"check_frequency": 10,    # Frequency of checking early stopping condition
	"minimum_epochs": 1000,   # Minimum epochs before early stopping can trigger
	"improvement_threshold": 0.00005,  # Minimum relative improvement required to reset patience
	"use_gradient_clipping": true,
	"gradient_clip_value": 0.8
}




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

	# Randomly choose the number of hidden layers (between 1 and 3)
	var num_hidden_layers = randi_range(1, 3)

	nnas.add_layer(6, nnas.ACTIVATIONS.LINEAR)
	#for i in range(num_hidden_layers):
		## Randomly choose the number of neurons for this layer (between 1 and 10)
		#var num_neurons = randi_range(3, 6)
		## Randomly choose an activation function for this layer
		#var activation_function_name = activation_functions.pick_random()
		#var activation_function = nnas.ACTIVATIONS[activation_function_name]
		## Add the layer to the network
		#nnas.add_layer(num_neurons, activation_function)

	# Output layer with 3 neurons (for RGB values)
	nnas.add_layer(3, nnas.ACTIVATIONS.LINEAR)
	nnas.load(nnas.save_path)
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

	# Update the ColorRect objects to visualize the input and expected output
	input.color = Color(normalized_input[0], normalized_input[1], normalized_input[2])
	var prediction = nnas.predict(normalized_input)
	output.color = Color(prediction[0], prediction[1], prediction[2])

	# Train the network
	for i in range(5):
		nnas.train(normalized_input, normalized_output)
	
	# After training, visualize the network
	$VisualizeNet2.visualize(nnas)
