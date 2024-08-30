class_name NeuralNetworkAdvanced

var network: Array

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var save_path: String = "res://nn.best.data"

var learning_rate: float = 0.01
var use_l2_regularization: bool = false
var l2_regularization_strength: float = 0.001
var use_adam_optimizer: bool = false  # Controls whether Adam optimizer is used
var beta1: float = 0.9
var beta2: float = 0.999
var epsilon: float = 1e-7

var layer_structure = []

var raycasts: Array[RayCast2D]

# Adam optimizer state
var momentums: Array = []
var velocities: Array = []
var t: int = 0

var use_gradient_clipping: bool = true  # Enable or disable gradient clipping
var gradient_clip_value: float = 1.0    # The maximum absolute value for the gradients

# Early Stopping Parameters
var early_stopping: bool = true
var has_stopped: bool = false
var minimum_epochs: int = 100  # Minimum number of epochs before early stopping can trigger
var patience: int = 50  # Increased patience for early stopping
var improvement_threshold: float = 0.005  # Relative improvement threshold

var best_loss: float = INF
var epochs_without_improvement: int = 0

var loss_history: Array = []
var smoothing_window: int = 10  # Number of epochs to average over
var check_frequency: int = 5  # Frequency of checking early stopping condition
var steps_completed: int = 0  # Tracks the number of steps/epochs completed


func _init(config: Dictionary):
	set_config(config)

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
	minimum_epochs = config.get("minimum_epochs", minimum_epochs)  # Add minimum_epochs to the config
	improvement_threshold = config.get("improvement_threshold", improvement_threshold)  # Add relative improvement threshold to the config
	# Gradient Clipping
	use_gradient_clipping = config.get("use_gradient_clipping", use_gradient_clipping)
	gradient_clip_value = config.get("gradient_clip_value", gradient_clip_value)

func add_layer(nodes: int, activation: Dictionary = ACTIVATIONS.SIGMOID):
	if layer_structure.size() != 0:
		var weights: Matrix = Matrix.new()
		var bias: Matrix = Matrix.new()
		
		weights.init(nodes, layer_structure[-1])
		bias.init(nodes, 1)
		weights.rand()
		bias.rand()

		var layer_data: Dictionary = {
			"weights": weights,
			"bias": bias,
			"activation": activation
		}

		# Initialize momentum and velocity for Adam optimizer only if enabled
		if use_adam_optimizer:
			var momentum_weights: Matrix = Matrix.new()
			momentum_weights.init(nodes, layer_structure[-1])  # Initializes with zeros
			var momentum_bias: Matrix = Matrix.new()
			momentum_bias.init(nodes, 1)  # Initializes with zeros
			
			var velocity_weights: Matrix = Matrix.new()
			velocity_weights.init(nodes, layer_structure[-1])  # Initializes with zeros
			var velocity_bias: Matrix = Matrix.new()
			velocity_bias.init(nodes, 1)  # Initializes with zeros

			var momentum: Dictionary = {
				"weights": momentum_weights,
				"bias": momentum_bias
			}
			var velocity: Dictionary = {
				"weights": velocity_weights,
				"bias": velocity_bias
			}
			
			momentums.append(momentum)
			velocities.append(velocity)

		network.push_back(layer_data)
	layer_structure.append(nodes)

func update_loss_history(current_loss: float):
	loss_history.append(current_loss)
	if loss_history.size() > smoothing_window:
		loss_history.remove_at(0)

func get_smoothed_loss() -> float:
	var smoothed_loss: float = 0.0
	for loss in loss_history:
		smoothed_loss += loss
	return smoothed_loss / loss_history.size()


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
	var outputs: Array[Matrix] = []
	var unactivated_outputs: Array[Matrix] = []
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, layer_inputs)
		var sum: Matrix = Matrix.add(product, layer.bias)
		var map: Matrix = Matrix.map(sum, layer.activation.function)
		layer_inputs = map
		outputs.append(map)
		unactivated_outputs.append(sum)
	
	var expected_output: Matrix = targets
	
	var next_layer_errors: Matrix
	
	for layer_index in range(network.size() - 1, -1, -1):
		var layer: Dictionary = network[layer_index]
		var layer_outputs: Matrix = outputs[layer_index]
		var layer_unactivated_output: Matrix = Matrix.transpose(unactivated_outputs[layer_index])

		if layer_index == network.size() - 1:
			var output_errors: Matrix = Matrix.subtract(expected_output, layer_outputs)
			next_layer_errors = output_errors
			var gradients: Matrix = Matrix.map(layer_outputs, layer.activation.derivative)
			gradients = Matrix.multiply(gradients, output_errors)
			gradients = Matrix.scalar(gradients, learning_rate)
			
			var weight_delta: Matrix
			
			if layer_index == 0:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(inputs))
			else:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(outputs[layer_index - 1]))
				  # Apply gradient clipping
			if use_gradient_clipping:
				gradients = clip_gradients(gradients)
				weight_delta = clip_gradients(weight_delta)
			if use_adam_optimizer:
				# Adam optimizer update
				t += 1
				var m: Dictionary = momentums[layer_index]
				var v: Dictionary = velocities[layer_index]
				
				m.weights = Matrix.add(Matrix.scalar(m.weights, beta1), Matrix.scalar(weight_delta, 1 - beta1))
				m.bias = Matrix.add(Matrix.scalar(m.bias, beta1), Matrix.scalar(gradients, 1 - beta1))
				
				v.weights = Matrix.add(Matrix.scalar(v.weights, beta2), Matrix.scalar(square_matrix(weight_delta), 1 - beta2))
				v.bias = Matrix.add(Matrix.scalar(v.bias, beta2), Matrix.scalar(square_matrix(gradients), 1 - beta2))
				
				var m_hat_weights = divide_matrix_by_scalar(m.weights, (1 - pow(beta1, t)))
				var m_hat_bias = divide_matrix_by_scalar(m.bias, (1 - pow(beta1, t)))
				
				var v_hat_weights = divide_matrix_by_scalar(v.weights, (1 - pow(beta2, t)))
				var v_hat_bias = divide_matrix_by_scalar(v.bias, (1 - pow(beta2, t)))
				
				# Element-wise division using Hadamard product and reciprocal
				weight_delta = multiply_elementwise(m_hat_weights, reciprocal(add_scalar_to_matrix(sqrt_matrix(v_hat_weights), epsilon)))
				gradients = multiply_elementwise(m_hat_bias, reciprocal(add_scalar_to_matrix(sqrt_matrix(v_hat_bias), epsilon)))
			
			# Update weights with L2 Regularization
			if use_l2_regularization:
				var l2_penalty: Matrix = Matrix.scalar(layer.weights, 2 * l2_regularization_strength)
				weight_delta = Matrix.subtract(weight_delta, l2_penalty)
			
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, gradients)
		else:
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
			
			# Apply gradient clipping
			if use_gradient_clipping:
				hidden_gradient = clip_gradients(hidden_gradient)
				weight_delta = clip_gradients(weight_delta)
			
			if use_adam_optimizer:
				# Adam optimizer update
				t += 1
				var m: Dictionary = momentums[layer_index]
				var v: Dictionary = velocities[layer_index]
				
				m.weights = Matrix.add(Matrix.scalar(m.weights, beta1), Matrix.scalar(weight_delta, 1 - beta1))
				m.bias = Matrix.add(Matrix.scalar(m.bias, beta1), Matrix.scalar(hidden_gradient, 1 - beta1))
				
				v.weights = Matrix.add(Matrix.scalar(v.weights, beta2), Matrix.scalar(square_matrix(weight_delta), 1 - beta2))
				v.bias = Matrix.add(Matrix.scalar(v.bias, beta2), Matrix.scalar(square_matrix(hidden_gradient), 1 - beta2))
				
				var m_hat_weights = divide_matrix_by_scalar(m.weights, (1 - pow(beta1, t)))
				var m_hat_bias = divide_matrix_by_scalar(m.bias, (1 - pow(beta1, t)))
				
				var v_hat_weights = divide_matrix_by_scalar(v.weights, (1 - pow(beta2, t)))
				var v_hat_bias = divide_matrix_by_scalar(v.bias, (1 - pow(beta2, t)))
				
				# Element-wise division using Hadamard product and reciprocal
				weight_delta = multiply_elementwise(m_hat_weights, reciprocal(add_scalar_to_matrix(sqrt_matrix(v_hat_weights), epsilon)))
				hidden_gradient = multiply_elementwise(m_hat_bias, reciprocal(add_scalar_to_matrix(sqrt_matrix(v_hat_bias), epsilon)))
			
			# Update weights with L2 Regularization
			if use_l2_regularization:
				var l2_penalty: Matrix = Matrix.scalar(layer.weights, 2 * l2_regularization_strength)
				weight_delta = Matrix.subtract(weight_delta, l2_penalty)
			
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, hidden_gradient)

	# After calculating the raw loss
	var loss: float = calculate_loss(targets, outputs[-1])

	# Add the loss to the history
	loss_history.append(loss)

	# Calculate the smoothed loss using moving average
	var smoothed_loss: float = calculate_moving_average(loss_history, smoothing_window)

# Early Stopping Logic using smoothed loss
	if early_stopping and steps_completed >= minimum_epochs:
		# Check if the smoothed loss is an improvement over the best recorded loss
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
			self.load(save_path)  # Restore the best model state
 # Restore the best model state

	steps_completed += 1
	return not has_stopped


func clip_gradients(gradients: Matrix) -> Matrix:
	var clipped_gradients: Matrix = Matrix.new()
	clipped_gradients.init(gradients.get_rows(), gradients.get_cols())
	
	for i in range(gradients.get_rows()):
		for j in range(gradients.get_cols()):
			var value = gradients.get_at(i, j)
			clipped_gradients.set_at(i, j, clamp(value, -gradient_clip_value, gradient_clip_value))
	
	return clipped_gradients


func calculate_loss(targets: Matrix, predictions: Matrix) -> float:
	var error: Matrix = Matrix.subtract(targets, predictions)
	var loss: float = 0.0
	for i in range(error.get_rows()):
		for j in range(error.get_cols()):
			loss += pow(error.get_at(i, j), 2)
	return loss / error.get_rows()

func calculate_moving_average(values: Array, window_size: int) -> float:
	var moving_sum: float = 0.0
	for i in range(max(0, values.size() - window_size), values.size()):
		moving_sum += values[i]
	return moving_sum / min(values.size(), window_size)


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

# Utility functions for squaring, square rooting a matrix, adding a scalar, and dividing by a scalar
func square_matrix(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix.get_rows(), matrix.get_cols())
	for i in range(matrix.get_rows()):
		for j in range(matrix.get_cols()):
			var value = matrix.get_at(i, j)
			result.set_at(i, j, value * value)
	return result

func sqrt_matrix(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix.get_rows(), matrix.get_cols())
	for i in range(matrix.get_rows()):
		for j in range(matrix.get_cols()):
			var value = matrix.get_at(i, j)
			result.set_at(i, j, sqrt(value))
	return result

# Add a scalar to each element in the matrix
func add_scalar_to_matrix(matrix: Matrix, scalar: float) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix.get_rows(), matrix.get_cols())
	for i in range(matrix.get_rows()):
		for j in range(matrix.get_cols()):
			result.set_at(i, j, matrix.get_at(i, j) + scalar)
	return result

# Divide each element in the matrix by a scalar
func divide_matrix_by_scalar(matrix: Matrix, scalar: float) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix.get_rows(), matrix.get_cols())
	for i in range(matrix.get_rows()):
		for j in range(matrix.get_cols()):
			result.set_at(i, j, matrix.get_at(i, j) / scalar)
	return result

# Element-wise multiplication of two matrices
func multiply_elementwise(matrix1: Matrix, matrix2: Matrix) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix1.get_rows(), matrix1.get_cols())
	for i in range(matrix1.get_rows()):
		for j in range(matrix1.get_cols()):
			result.set_at(i, j, matrix1.get_at(i, j) * matrix2.get_at(i, j))
	return result

# Reciprocal of each element in the matrix
func reciprocal(matrix: Matrix) -> Matrix:
	var result: Matrix = Matrix.new()
	result.init(matrix.get_rows(), matrix.get_cols())
	for i in range(matrix.get_rows()):
		for j in range(matrix.get_cols()):
			result.set_at(i, j, 1.0 / matrix.get_at(i, j))
	return result

func get_inputs_from_raycasts() -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _input_array: Array[float]
	
	for ray in raycasts:
		if is_instance_valid(ray): _input_array.push_front(get_distance(ray))
	
	return _input_array

func get_prediction_from_raycasts(optional_val: Array = []) -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _array_ = get_inputs_from_raycasts()
	_array_.append_array(optional_val)
	return predict(_array_)

func get_distance(_raycast: RayCast2D):
	var distance: float = 0.0
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()
		
		distance = origin.distance_to(collision)
	else:
		distance = sqrt((pow(_raycast.target_position.x, 2) + pow(_raycast.target_position.y, 2)))
	return distance

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
