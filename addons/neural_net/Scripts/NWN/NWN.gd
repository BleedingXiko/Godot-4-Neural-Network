class_name NeuralNetworkInNetwork

var af = Activation.new()
var ACTIVATIONS = af.get_functions()

var sub_networks: Array
var master_network: NeuralNetworkAdvanced

var network_config = {
	"learning_rate": 0.000001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

func _init(master_config: Dictionary):
	sub_networks = []
	master_network = NeuralNetworkAdvanced.new(master_config)

func add_nin_layer(neurons_count: int, input_size: int, sub_hidden_layers: Array = []):
	# Create a layer where each "neuron" is a small neural network
	var layer = []
	for i in range(neurons_count):
		var sub_nn = NeuralNetworkAdvanced.new(network_config)
		sub_nn.add_layer(input_size)  # Input layer for the sub-neuron
		for hidden_layer in sub_hidden_layers:
			sub_nn.add_layer(hidden_layer, ACTIVATIONS.SIGMOID)  # Hidden layers for the sub-neuron
		sub_nn.add_layer(1, ACTIVATIONS.LINEAR)  # Output layer for the sub-neuron
		layer.append(sub_nn)
	sub_networks.append(layer)

func add_master_network_layer(output_size: int, master_hidden_layers: Array = [], activation: Dictionary = ACTIVATIONS.SIGMOID):
	# The master network will have hidden layers and an output layer
	if sub_networks.size() > 0:
		var input_size = sub_networks[-1].size()  # Number of sub-networks in the last layer
		master_network.add_layer(input_size)  # Master network's input layer
		for hidden_layer in master_hidden_layers:
			master_network.add_layer(hidden_layer, activation)  # Hidden layers for the master network
		master_network.add_layer(output_size, activation)  # Master network's output layer

func predict(input_array: Array) -> Array:
	var inputs = input_array
	for layer in sub_networks:
		var layer_output = []
		for sub_nn in layer:
			# Get the output from each sub-network (which acts as a neuron)
			var sub_output = sub_nn.predict(inputs)
			layer_output.append(sub_output[0])
		inputs = layer_output

	# Pass the combined output of all sub-networks to the master network
	return master_network.predict(inputs)

func train(input_array: Array, target_array: Array):
	var inputs = input_array
	var outputs: Array = []

	# Forward pass through sub-networks
	for layer in sub_networks:
		var layer_output = []
		for sub_nn in layer:
			var sub_output = sub_nn.predict(inputs)
			layer_output.append(sub_output[0])
		outputs.append(layer_output)
		inputs = layer_output

	# Ensure the output from sub-networks matches the input size of the master network
	var combined_output = outputs[-1]
	if combined_output.size() != master_network.layer_structure[0]:
		print("Mismatch between sub-network output and master network input.")
		return

	# Train the master network with the combined outputs of sub-networks
	master_network.train(combined_output, target_array)

	# Backpropagation through sub-networks
	for i in range(sub_networks.size() - 1, -1, -1):
		var layer = sub_networks[i]
		var layer_output = outputs[i]
		var previous_output = input_array if i == 0 else outputs[i - 1]

		for j in range(layer.size()):
			var sub_nn = layer[j]
			var sub_target = [layer_output[j]]

			# Train the sub-network with the output from the previous layer
			sub_nn.train(previous_output, sub_target)

func copy() -> NeuralNetworkInNetwork:
	var new_network = NeuralNetworkInNetwork.new(network_config)
	for layer in sub_networks:
		var new_layer = []
		for sub_nn in layer:
			var copied_sub_nn = sub_nn.copy()
			new_layer.append(copied_sub_nn)
		new_network.sub_networks.append(new_layer)
	new_network.master_network = master_network.copy()
	return new_network

func save(path: String):
	# Save master network
	master_network.save(path + "_master")
	# Save each sub-network
	for i in range(sub_networks.size()):
		for j in range(sub_networks[i].size()):
			sub_networks[i][j].save(path + "_sub_" + str(i) + "_" + str(j))

func load(path: String, config: Dictionary):
	# Load master network
	master_network.load(path + "_master")
	# Load each sub-network
	for i in range(sub_networks.size()):
		for j in range(sub_networks[i].size()):
			sub_networks[i][j].load(path + "_sub_" + str(i) + "_" + str(j))
	set_config(config)

func set_config(config: Dictionary):
	network_config = config
	master_network.set_config(config)
	for layer in sub_networks:
		for sub_nn in layer:
			sub_nn.set_config(config)
