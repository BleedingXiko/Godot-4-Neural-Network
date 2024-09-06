class_name NeuralNetworkAdvanced

# Variables for the neural network structure
var network: Array  # Stores the layers of the neural network
var af = Activation.new()  # Instance of activation functions
var ACTIVATIONS = af.get_functions()  # Dictionary of available activation functions

# File path to save the best model
var save_path: String = "res://nn.best.data"

# Learning parameters
var learning_rate: float = 0.01  # Standard learning rate for SGD
var use_l2_regularization: bool = false  # L2 regularization off by default
var l2_regularization_strength: float = 0.001  # If L2 is enabled

##TODO ADAM DOESNT WORK
var use_adam_optimizer: bool = false  # Use SGD by default, not Adam
var beta1: float = 0.9  # Adam's first moment decay rate (only used if Adam is enabled)
var beta2: float = 0.999  # Adam's second moment decay rate
var epsilon: float = 1e-7  # Adam's epsilon for numerical stability

# Structure of the neural network layers
var layer_structure = []

# Adam optimizer state variables (only used if Adam is enabled)
var momentums: Array = []  
var velocities: Array = []  
var t: int = 0  # Time step counter for Adam optimizer

# Gradient clipping parameters
var use_gradient_clipping: bool = false  # Disabled by default, only enable if necessary
var gradient_clip_value: float = 1.0  # Clipping value if enabled

# Early stopping parameters
var early_stopping: bool = false  # Early stopping disabled by default
var has_stopped: bool = false  # Flag indicating if early stopping has been triggered
var minimum_epochs: int = 100  # Default number of minimum epochs if early stopping is enabled
var patience: int = 50  # Default patience for early stopping if enabled
var improvement_threshold: float = 0.005  # Threshold for improvement in early stopping

# Variables to track progress for early stopping
var best_loss: float = INF  # Best recorded loss
var epochs_without_improvement: int = 0  # Counter for epochs without improvement

# Initialization method for weights
var initialization_type: String = "random"  # Random initialization by default

# Loss history tracking for smoothing and early stopping
var loss_history: Array = []  # Array to store loss values for each epoch
var smoothing_window: int = 10  # Number of epochs to average over for smoothing
var check_frequency: int = 5  # Frequency of checking for early stopping conditions
var steps_completed: int = 0  # Total number of epochs/steps completed

# Add this variable to hold the loss function type
var loss_function_type: String = "mse"  # Default to MSE

# Initialization function
func _init(config: Dictionary):
	set_config(config)  # Set configuration from a dictionary

# Function to set network configuration from a dictionary
func set_config(config: Dictionary):
	learning_rate = config.get("learning_rate", learning_rate)
	use_l2_regularization = config.get("use_l2_regularization", use_l2_regularization)
	l2_regularization_strength = config.get("l2_regularization_strength", l2_regularization_strength)
	use_adam_optimizer = config.get("use_adam_optimizer", use_adam_optimizer)
	beta1 = config.get("beta1", beta1)
	beta2 = config.get("beta2", beta2)
	epsilon = config.get("epsilon", epsilon)
	early_stopping = config.get("early_stopping", early_stopping)
	patience = config.get("patience", patience)
	save_path = config.get("save_path", save_path)
	smoothing_window = config.get("smoothing_window", smoothing_window)
	check_frequency = config.get("check_frequency", check_frequency)
	minimum_epochs = config.get("minimum_epochs", minimum_epochs)
	improvement_threshold = config.get("improvement_threshold", improvement_threshold)
	gradient_clip_value = config.get("gradient_clip_value", gradient_clip_value)
	initialization_type = config.get("initialization_type", initialization_type)
	loss_function_type = config.get("loss_function_type", loss_function_type)  # Add loss function to config

# Function to add a new layer to the network
func add_layer(nodes: int, activation: Dictionary = ACTIVATIONS.SIGMOID):
	if layer_structure.size() != 0:
		var weights: Matrix = Matrix.new()
		var bias: Matrix = Matrix.new()
		
		weights.init(nodes, layer_structure[-1])
		bias.init(nodes, 1)

		if initialization_type == "xavier":
			weights.init_xavier(nodes, layer_structure[-1])
		elif initialization_type == "he":
			weights.init_he(nodes, layer_structure[-1])
		else:
			weights.rand()

		var layer_data: Dictionary = {
			"weights": weights,
			"bias": bias,
			"activation": activation
		}

		# Initialize momentum for all optimizers
		var momentum_weights: Matrix = Matrix.new()
		var momentum_bias: Matrix = Matrix.new()
		momentum_weights.init(nodes, layer_structure[-1])
		momentum_bias.init(nodes, 1)

		var momentum: Dictionary = {
			"weights": momentum_weights,
			"bias": momentum_bias
		}
		momentums.append(momentum)

		# Initialize velocity only if using Adam optimizer
		if use_adam_optimizer:
			var velocity_weights: Matrix = Matrix.new()
			var velocity_bias: Matrix = Matrix.new()
			velocity_weights.init(nodes, layer_structure[-1])
			velocity_bias.init(nodes, 1)

			var velocity: Dictionary = {
				"weights": velocity_weights,
				"bias": velocity_bias
			}
			
			velocities.append(velocity)
		# Add the layer to the network
		network.push_back(layer_data)

	layer_structure.append(nodes)

# Function to make predictions based on input data
func predict(input_array: Array) -> Array:
	var inputs: Matrix = Matrix.from_array(input_array)
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, inputs)
		var sum: Matrix = Matrix.add(product, layer.bias)
		var map: Matrix = Matrix.map(sum, layer.activation.function)
		inputs = map
	return Matrix.to_array(inputs)

func train(input_array: Array, target_array: Array) -> bool:
	if has_stopped:
		return false

	var inputs: Matrix = Matrix.from_array(input_array)
	var targets: Matrix = Matrix.from_array(target_array)

	var layer_inputs: Matrix = inputs
	var outputs: Array[Matrix] = []  # Store activated outputs for each layer
	var unactivated_outputs: Array[Matrix] = []  # Store unactivated outputs for each layer

	# Forward pass
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, layer_inputs)
		var sum: Matrix = Matrix.add(product, layer.bias)
		var map: Matrix = Matrix.map(sum, layer.activation.function)
		layer_inputs = map
		outputs.append(map)
		unactivated_outputs.append(sum)

	var expected_output: Matrix = targets
	var next_layer_errors: Matrix

	# Increment t once per training step (persist across epochs)
	t += 1

	# Backward pass
	for layer_index in range(network.size() - 1, -1, -1):
		var layer: Dictionary = network[layer_index]
		var layer_outputs: Matrix = outputs[layer_index]
		var layer_unactivated_output: Matrix = Matrix.transpose(unactivated_outputs[layer_index])

		if layer_index == network.size() - 1:  # Output layer
			var output_errors: Matrix
			match loss_function_type:
				"mse":
					output_errors = Matrix.mse_gradient(layer_outputs, expected_output)
				"cross_entropy":
					output_errors = Matrix.cross_entropy_gradient(layer_outputs, expected_output)
				"binary_cross_entropy":
					output_errors = Matrix.binary_cross_entropy_gradient(layer_outputs, expected_output)
				"huber_loss":
					output_errors = Matrix.huber_gradient(layer_outputs, expected_output)
				_:
					output_errors = Matrix.mse_gradient(layer_outputs, expected_output)

			next_layer_errors = output_errors
			var gradients: Matrix = Matrix.map(layer_outputs, layer.activation.derivative)
			gradients = Matrix.multiply(gradients, output_errors)
			gradients = Matrix.scalar(gradients, learning_rate)

			var weight_delta: Matrix
			if layer_index == 0:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(inputs))
			else:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(outputs[layer_index - 1]))

			# L2 regularization: Adding the regularization term to weight_delta
			if use_l2_regularization:
				var l2_penalty_weights: Matrix = Matrix.scalar(layer.weights, l2_regularization_strength)
				weight_delta = Matrix.subtract(weight_delta, l2_penalty_weights)

			# Gradient clipping (if enabled)
			if use_gradient_clipping:
				gradients = Matrix.clip_gradients(gradients, gradient_clip_value)
				weight_delta = Matrix.clip_gradients(weight_delta, gradient_clip_value)

			# Adam Optimizer Logic
			if use_adam_optimizer:
				var m: Dictionary = momentums[layer_index]
				var v: Dictionary = velocities[layer_index]

				# Update Adam's momentum (m) and velocity (v)
				m.weights = Matrix.add(Matrix.scalar(m.weights, beta1), Matrix.scalar(weight_delta, 1 - beta1))
				m.bias = Matrix.add(Matrix.scalar(m.bias, beta1), Matrix.scalar(gradients, 1 - beta1))

				v.weights = Matrix.add(Matrix.scalar(v.weights, beta2), Matrix.scalar(Matrix.square_matrix(weight_delta), 1 - beta2))
				v.bias = Matrix.add(Matrix.scalar(v.bias, beta2), Matrix.scalar(Matrix.square_matrix(gradients), 1 - beta2))

				# Bias correction
				var m_hat_weights = Matrix.divide_matrix_by_scalar(m.weights, (1 - pow(beta1, t)))
				var m_hat_bias = Matrix.divide_matrix_by_scalar(m.bias, (1 - pow(beta1, t)))
				var v_hat_weights = Matrix.divide_matrix_by_scalar(v.weights, (1 - pow(beta2, t)))
				var v_hat_bias = Matrix.divide_matrix_by_scalar(v.bias, (1 - pow(beta2, t)))

				# Update weights and biases using Adam's update rule
				weight_delta = Matrix.multiply(m_hat_weights, Matrix.reciprocal(Matrix.add_scalar_to_matrix(Matrix.sqrt_matrix(v_hat_weights), epsilon)))
				gradients = Matrix.multiply(m_hat_bias, Matrix.reciprocal(Matrix.add_scalar_to_matrix(Matrix.sqrt_matrix(v_hat_bias), epsilon)))

				# Apply the Adam update to the weights and biases
				network[layer_index].weights = Matrix.subtract(layer.weights, weight_delta)
				network[layer_index].bias = Matrix.subtract(layer.bias, gradients)
			else:
				# Fallback to SGD with Momentum (if Adam isn't being used)
				var momentum_coefficient: float = 0.9
				var m: Dictionary = momentums[layer_index]

				# Update momentum using the SGD momentum rule
				m.weights = Matrix.add(Matrix.scalar(m.weights, momentum_coefficient), weight_delta)
				m.bias = Matrix.add(Matrix.scalar(m.bias, momentum_coefficient), gradients)

				# Use momentum to update weights and biases
				network[layer_index].weights = Matrix.subtract(layer.weights, m.weights)
				network[layer_index].bias = Matrix.subtract(layer.bias, m.bias)

		else:  # Hidden layers (similar logic to output layer)
			var weights_hidden_output_t = Matrix.transpose(network[layer_index + 1].weights)
			var hidden_errors = Matrix.dot_product(weights_hidden_output_t, next_layer_errors)
			next_layer_errors = hidden_errors

			var hidden_gradient = Matrix.map(layer_outputs, layer.activation.derivative)
			hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
			hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate)

			var inputs_t: Matrix
			if layer_index != 0:
				inputs_t = Matrix.transpose(outputs[layer_index - 1])
			else:
				inputs_t = Matrix.transpose(inputs)

			var weight_delta = Matrix.dot_product(hidden_gradient, inputs_t)

			# L2 regularization for hidden layers
			if use_l2_regularization:
				var l2_penalty_weights_hidden: Matrix = Matrix.scalar(layer.weights, l2_regularization_strength)
				weight_delta = Matrix.subtract(weight_delta, l2_penalty_weights_hidden)

			# Gradient clipping (if enabled)
			if use_gradient_clipping:
				hidden_gradient = Matrix.clip_gradients(hidden_gradient, gradient_clip_value)
				weight_delta = Matrix.clip_gradients(weight_delta, gradient_clip_value)

			# Adam optimizer for hidden layers
			if use_adam_optimizer:
				var m: Dictionary = momentums[layer_index]
				var v: Dictionary = velocities[layer_index]

				m.weights = Matrix.add(Matrix.scalar(m.weights, beta1), Matrix.scalar(weight_delta, 1 - beta1))
				m.bias = Matrix.add(Matrix.scalar(m.bias, beta1), Matrix.scalar(hidden_gradient, 1 - beta1))

				v.weights = Matrix.add(Matrix.scalar(v.weights, beta2), Matrix.scalar(Matrix.square_matrix(weight_delta), 1 - beta2))
				v.bias = Matrix.add(Matrix.scalar(v.bias, beta2), Matrix.scalar(Matrix.square_matrix(hidden_gradient), 1 - beta2))

				var m_hat_weights = Matrix.divide_matrix_by_scalar(m.weights, (1 - pow(beta1, t)))
				var m_hat_bias = Matrix.divide_matrix_by_scalar(m.bias, (1 - pow(beta1, t)))
				var v_hat_weights = Matrix.divide_matrix_by_scalar(v.weights, (1 - pow(beta2, t)))
				var v_hat_bias = Matrix.divide_matrix_by_scalar(v.bias, (1 - pow(beta2, t)))

				weight_delta = Matrix.multiply(m_hat_weights, Matrix.reciprocal(Matrix.add_scalar_to_matrix(Matrix.sqrt_matrix(v_hat_weights), epsilon)))
				hidden_gradient = Matrix.multiply(m_hat_bias, Matrix.reciprocal(Matrix.add_scalar_to_matrix(Matrix.sqrt_matrix(v_hat_bias), epsilon)))

				# Apply Adam updates to the weights and biases
				network[layer_index].weights = Matrix.subtract(layer.weights, weight_delta)
				network[layer_index].bias = Matrix.subtract(layer.bias, hidden_gradient)

			else:
				# SGD with momentum for hidden layers
				var momentum_coefficient: float = 0.9  # Momentum coefficient for SGD
				var m: Dictionary = momentums[layer_index]

				m.weights = Matrix.add(Matrix.scalar(m.weights, momentum_coefficient), weight_delta)
				m.bias = Matrix.add(Matrix.scalar(m.bias, momentum_coefficient), hidden_gradient)

				# Use momentum to update weights and biases
				network[layer_index].weights = Matrix.subtract(layer.weights, m.weights)
				network[layer_index].bias = Matrix.subtract(layer.bias, m.bias)

	# Early stopping logic
	var loss: float
	match loss_function_type:
		"mse":
			loss = Matrix.mse_loss(targets, outputs[-1])
		"cross_entropy":
			loss = Matrix.cross_entropy_loss(targets, outputs[-1])
		"binary_cross_entropy":
			loss = Matrix.binary_cross_entropy_loss(targets, outputs[-1])
		"huber_loss":
			loss = Matrix.huber_loss(targets, outputs[-1])
		_:
			loss = Matrix.mse_loss(targets, outputs[-1])  # Default to MSE

	# Update the loss history
	loss_history.append(loss)

	# Calculate the smoothed loss using a moving average
	var smoothed_loss: float = calculate_moving_average(loss_history, smoothing_window)

	# Early Stopping Logic
	if early_stopping and steps_completed >= minimum_epochs:
		if best_loss == INF or (best_loss - smoothed_loss) / abs(best_loss) > improvement_threshold:
			best_loss = smoothed_loss
			epochs_without_improvement = 0
			self.save(save_path)
			print("Model saved at epoch:", steps_completed, "with smoothed loss:", smoothed_loss)
		else:
			epochs_without_improvement += 1
			print("No significant improvement. Epochs without improvement:", epochs_without_improvement)

		if epochs_without_improvement >= patience:
			has_stopped = true
			print("Early stopping triggered. Restoring best model saved with loss:", best_loss)
			self.load(save_path)

	steps_completed += 1
	return not has_stopped  # Continue training if early stopping hasn't been triggered

# Function to calculate a moving average over a specified window size
func calculate_moving_average(values: Array, window_size: int) -> float:
	var moving_sum: float = 0.0
	for i in range(max(0, values.size() - window_size), values.size()):
		moving_sum += values[i]
	return moving_sum / min(values.size(), window_size)

# Function to create a copy of the neural network with the same structure and parameters
func copy() -> NeuralNetworkAdvanced:
	# Copy other necessary properties if there are any
	var new_network = NeuralNetworkAdvanced.new(
		{
		"learning_rate": learning_rate,
		"l2_regularization_strength": l2_regularization_strength,
		"use_l2_regularization": use_l2_regularization,
		"use_adam_optimizer": use_adam_optimizer,
		"beta1": beta1,
		"beta2": beta2,
		"epsilon": epsilon
		})
		
	for layer in network:
		var layer_copy: Dictionary = {
			"weights": Matrix.copy(layer.weights),  # Copy weights
			"bias": Matrix.copy(layer.bias),       # Copy biases
			"activation": layer.activation        # Copy activation function
			}
		new_network.network.push_back(layer_copy)
		
	new_network.layer_structure = layer_structure.duplicate()

	return new_network

# Function to save the model to a file
func save(path: String):
	var file = FileAccess.open(path, FileAccess.WRITE)
	var data_to_save = []
	for layer in network:
		var layer_data = {
			"weights": layer.weights.save(),
			"bias": layer.bias.save(),
			"activation": layer.activation.name
		}
		data_to_save.append(layer_data)
	
	file.store_var(data_to_save)
	file.close()
	print(data_to_save)

# Function to return the network structure for debugging purposes
func debug():
	var data = []
	for layer in network:
		var layer_data = {
			"weights": layer.weights.save(),
			"bias": layer.bias.save(),
			"activation": layer.activation.name
		}
		data.append(layer_data)
	return data

# Function to load the model from a file
func load(path: String):
	var file = FileAccess.open(path, FileAccess.READ)
	var data = file.get_var()
	
	network.clear()
	for layer_data in data:
		var layer = {
			"weights": Matrix.load(layer_data.weights),
			"bias": Matrix.load(layer_data.bias),
			"activation": ACTIVATIONS[layer_data.activation]
		}
		network.append(layer)
	file.close()
