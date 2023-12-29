extends Node2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
var nn: NeuralNetworkAdvanced
var data = []

var numbers = [1,2,3,4,5,6,7,8,9,0]

var training_data = [{
			"inputs": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
,
			"target": [1,0,0,0,0,0,0,0,0,0]
}, 
{
			"inputs": [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]
,
			"target": [1,0,0,0,0,0,0,0,0,0]
}, 
{
			"inputs": [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
,
			"target": [0,1,0,0,0,0,0,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]
,
			"target": [0,0,1,0,0,0,0,0,0,0]
},
{
			"inputs": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

,
			"target": [0,0,0,1,0,0,0,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,1,0,0,0,0,0]
},
{
			"inputs": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,0,1,0,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,0,1,0,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


,
			"target": [0,0,0,0,0,0,1,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]


,
			"target": [0,0,0,0,0,0,1,0,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,0,0,0,1,0,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

,
			"target": [0,0,0,0,0,0,0,0,1,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,0,0,0,0,1,0]
},
{
			"inputs": [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]

,
			"target": [0,0,0,0,0,0,0,0,0,1]
}]

var network_config = {
	"learning_rate": 0.00000001,
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}
# Called when the node enters the scene tree for the first time.
func _ready():
	nn = NeuralNetworkAdvanced.new(network_config)
	
	nn.add_layer(25)
	nn.add_layer(26, nn.ACTIVATIONS.RELU)
	nn.add_layer(16, nn.ACTIVATIONS.RELU)
	nn.add_layer(10, nn.ACTIVATIONS.SIGMOID)
	
	#nn.load("./nn.data")
	test()
#	for n in get_children():
#		data.append(n.data)
#	print(data)

func test():
	for i in range(20000):
		var data = training_data.pick_random()
		nn.train(data.inputs, data.target)
	nn.save("./nn.data")


func findMax(arr):
	var record = 0
	var index = 0
	for i in arr.size():
		if arr[i] > record:
			record = arr[i]
			index = i
	
	return index

func _input(event):
	if event.is_action_pressed("ui_down"):
		nn.save("./nn.data")
	if event.is_action_pressed("ui_accept"):
		#nn.save_brain()
		data = []
		for n in $cells.get_children():
			data.append(n.data)
		
		
		var guess = numbers[findMax(nn.predict(data))]
		$Label2D.text = str(guess)
		print(nn.predict(data))

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
