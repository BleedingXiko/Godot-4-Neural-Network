extends Node2D

var nnas: NeuralNetworkAdvanced

var network_config = {
	"learning_rate": 0.0001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": true,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7
}

func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new(network_config)
	nnas.add_layer(3)
	nnas.add_layer(6, nnas.ACTIVATIONS.LINEAR)
	nnas.add_layer(4, nnas.ACTIVATIONS.LINEAR)
	nnas.add_layer(1, nnas.ACTIVATIONS.LINEAR)
	
	
	#for i in range(6000):
		#nnas.train([1.0, 2.0, 3.0], [6.0])
		#nnas.train([4.0, 5.0, 6.0], [15.0])
		#nnas.train([7.0, 8.0, 9.0], [24.0])
		#nnas.train([10.0, 11.0, 12.0], [33.0])
		#nnas.train([13.0, 14.0, 15.0], [42.0])


func _physics_process(delta: float) -> void:
	if Input.is_action_pressed("predict"):
		for i in range(1):
			# Generate random integer inputs
			var input1 = randi_range(1, 50)
			var input2 = randi_range(1, 50)
			var input3 = randi_range(1, 50)
			
			# The target is the sum of the inputs
			var target = input1 + input2 + input3
			
			# Train the network
			nnas.train([float(input1), float(input2), float(input3)], [float(target)])
		
		print("--------------Prediction--------------")
		print(nnas.predict([1.0, 2.0, 3.0]))
		print(nnas.predict([4.0, 5.0, 6.0]))
		print(nnas.predict([7.0, 8.0, 9.0]))
		print(nnas.predict([10.0, 11.0, 12.0]))
		print(nnas.predict([13.0, 14.0, 15.0]))
		print(nnas.predict([10.0, 10.0, 10.0]))
		$VisualizeNet.visualize(nnas)
