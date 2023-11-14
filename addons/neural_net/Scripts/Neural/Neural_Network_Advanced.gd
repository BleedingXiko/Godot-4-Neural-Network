class_name NeuralNetworkAdvanced

var network: Array

var ACTIVATIONS: Dictionary = {
	"SIGMOID": {
		"function": Callable(Activation, "sigmoid"),
		"derivative": Callable(Activation, "dsigmoid")
	},
	"RELU": {
		"function": Callable(Activation, "relu"),
		"derivative": Callable(Activation, "drelu")
	},
	"TANH": {
		"function": Callable(Activation, "tanh_"),
		"derivative": Callable(Activation, "dtanh")
	},
	"ARCTAN": {
		"function": Callable(Activation, "arcTan"),
		"derivative": Callable(Activation, "darcTan")
	},
	"PRELU": {
		"function": Callable(Activation, "prelu"),
		"derivative": Callable(Activation, "dprelu")
	},
	"ELU": {
		"function": Callable(Activation, "elu"),
		"derivative": Callable(Activation, "delu")
	},
	"SOFTPLUS": {
		"function": Callable(Activation, "softplus"),
		"derivative": Callable(Activation, "dsoftplus")
	}
}


var learning_rate: float = 0.5
var hidden_func: Dictionary = ACTIVATIONS.TANH 
var output_func: Dictionary = ACTIVATIONS.SIGMOID

var layer_structure = []

var raycasts: Array[RayCast2D]

func _init(a, b, c):
	
	add_layer(a)
	
	for i in b:
		add_layer(i, hidden_func)
		
	add_layer(c, output_func)

func add_layer(nodes: int, activation: Dictionary = ACTIVATIONS.SIGMOID):
	
	if layer_structure.size() != 0:
		var layer_data: Dictionary = {
			"weights": Matrix.rand(Matrix.new(nodes, layer_structure[-1])),
			"bias": Matrix.rand(Matrix.new(nodes, 1)),
			"activation": activation
		}
		network.push_back(layer_data)

	layer_structure.append(nodes)


func predict(input_array: Array) -> Array:
	var inputs: Matrix = Matrix.from_array(input_array)
	for layer in network:
		var product: Matrix = Matrix.dot_product(layer.weights, inputs)
		var sum: Matrix = Matrix.add(product, layer.bias)
		#print(layer.activation.function)
		var map: Matrix = Matrix.map(sum, layer.activation.function)
		inputs = map
	return Matrix.to_array(inputs)
#
func train(input_array: Array, target_array: Array):
	var inputs: Matrix = Matrix.from_array(input_array)
	var targets: Matrix = Matrix.from_array(target_array)

	var layer_inputs: Matrix = inputs
	var outputs: Array[Matrix]
	var unactivated_outputs: Array[Matrix]
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
#		print(layer_index)
		var layer: Dictionary = network[layer_index]
		var layer_outputs: Matrix = outputs[layer_index]
		var layer_unactivated_output: Matrix = Matrix.transpose(unactivated_outputs[layer_index])
#		print(layer_output.data)

		if layer_index == network.size() - 1:
			var output_errors: Matrix = Matrix.subtract(expected_output, layer_outputs)
			next_layer_errors = output_errors
	#		print(Matrix.map(error, layer.activation.derivative).data)
			var gradients: Matrix = Matrix.map(layer_outputs, layer.activation.derivative)
			gradients = Matrix.multiply(gradients, output_errors)
			gradients = Matrix.scalar(gradients, learning_rate)
			
#			print(outputs[layer_index - 1].data)
#			print(layer_unactivated_output.data)
#			print(layer.weights.data)
			var weight_delta: Matrix
#			print(layer_index)
			if layer_index == 0:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(inputs))
			else:
				weight_delta = Matrix.dot_product(gradients, Matrix.transpose(outputs[layer_index - 1]))
#			print(weight_delta.data)
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, gradients)
#			print("Success")
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
			
			network[layer_index].weights = Matrix.add(layer.weights, weight_delta)
			network[layer_index].bias = Matrix.add(layer.bias, hidden_gradient)
			

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
