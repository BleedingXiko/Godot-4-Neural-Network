extends Node2D


func _input(event):
	if event.is_action_pressed("ui_up"):
		$Neural_Net.best_nn.save("./nn.data")
	if event.is_action_pressed("ui_down"):
		$Neural_Net.best_nn.load("./nn.data")
