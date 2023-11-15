extends Node2D
var ACTIVATIONS: Dictionary = {
	"SIGMOID": {
		"function": Callable(Activation, "sigmoid"),
		"derivative": Callable(Activation, "dsigmoid"),
		"name": "SIGMOID",
	},
	"RELU": {
		"function": Callable(Activation, "relu"),
		"derivative": Callable(Activation, "drelu"),
		"name": "RELU"
	},
	"TANH": {
		"function": Callable(Activation, "tanh_"),
		"derivative": Callable(Activation, "dtanh"),
		"name": "TANH"
	},
	"ARCTAN": {
		"function": Callable(Activation, "arcTan"),
		"derivative": Callable(Activation, "darcTan"),
		"name": "ARCTAN"
	},
	"PRELU": {
		"function": Callable(Activation, "prelu"),
		"derivative": Callable(Activation, "dprelu"),
		"name": "PRELU"
	},
	"ELU": {
		"function": Callable(Activation, "elu"),
		"derivative": Callable(Activation, "delu"),
		"name": "ELU"
	},
	"SOFTPLUS": {
		"function": Callable(Activation, "softplus"),
		"derivative": Callable(Activation, "dsoftplus"),
		"name": "SOFTPLUS"
	}
}

var nnas: NeuralNetworkAdvanced

func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new(2, [6,3,2], 1, ACTIVATIONS.TANH, ACTIVATIONS.SIGMOID)
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
