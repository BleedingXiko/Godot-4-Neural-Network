extends Node2D

var nnas: NeuralNetworkAdvanced

var network_config = {
	"learning_rate": 0.0001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}

func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new(network_config)
	nnas.add_layer(3)
	nnas.add_layer(6, nnas.ACTIVATIONS.SIGMOID)
	nnas.add_layer(4, nnas.ACTIVATIONS.SIGMOID)
	nnas.add_layer(1, nnas.ACTIVATIONS.LINEAR)
	
	
	#for i in range(6000):
		#nnas.train([1.0, 2.0, 3.0], [6.0])
		#nnas.train([4.0, 5.0, 6.0], [15.0])
		#nnas.train([7.0, 8.0, 9.0], [24.0])
		#nnas.train([10.0, 11.0, 12.0], [33.0])
		#nnas.train([13.0, 14.0, 15.0], [42.0])


func _physics_process(delta: float) -> void:
	if Input.is_action_pressed("predict"):
		#nnas.load("./test.nn")
		for i in range(30):
			nnas.train([1.0, 2.0, 3.0], [6.0])
			nnas.train([4.0, 5.0, 6.0], [15.0])
			nnas.train([7.0, 8.0, 9.0], [24.0])
			nnas.train([10.0, 11.0, 12.0], [33.0])
			nnas.train([13.0, 14.0, 15.0], [42.0])
		print("--------------Prediction--------------")
		print(nnas.predict([1.0, 2.0, 3.0]))
		print(nnas.predict([4.0, 5.0, 6.0]))
		print(nnas.predict([7.0, 8.0, 9.0]))
		print(nnas.predict([10.0, 11.0, 12.0]))
		print(nnas.predict([13.0, 14.0, 15.0]))
		print(nnas.predict([10.0, 11.0, 13.0]))
		$VisualizeNet.visualize(nnas)
