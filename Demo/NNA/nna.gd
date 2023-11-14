extends Node2D

var nnas: NeuralNetworkAdvanced
func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new(2, [6,3,2], 1)
	nnas.hidden_func = nnas.ACTIVATIONS.TANH
	nnas.output_func = nnas.ACTIVATIONS.SIGMOID
	nnas.learning_rate = 0.01
	

func _physics_process(delta: float) -> void:
	nnas.train([0,0], [0])
	nnas.train([1,0], [1])
	nnas.train([0,1], [1])
	nnas.train([1,1], [0])
	
	if Input.is_action_just_pressed("predict"):
		print("--------------Prediction--------------")
		print(nnas.predict([0,0]))
		print(nnas.predict([1,0]))
		print(nnas.predict([0,1]))
		print(nnas.predict([1,1]))
