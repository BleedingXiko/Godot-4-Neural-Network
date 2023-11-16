extends Node2D

var nnas: NeuralNetworkAdvanced

func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new()
	nnas.add_layer(2)
	nnas.add_layer(6, nnas.ACTIVATIONS.TANH)
	nnas.add_layer(1, nnas.ACTIVATIONS.SIGMOID)
	nnas.learning_rate = 0.1
	nnas.use_l2_regularization = false
	nnas.l2_regularization_strength = 0.001
	
	
	for i in range(1200):
		nnas.train([0,0], [0])
		nnas.train([1,0], [1])
		nnas.train([0,1], [1])
		nnas.train([1,1], [0])


func _physics_process(delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		#nnas.save("./test.nn")
		print("--------------Prediction--------------")
		print(nnas.predict([0,0]))
		print(nnas.predict([1,0]))
		print(nnas.predict([0,1]))
		print(nnas.predict([1,1]))
