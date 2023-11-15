extends Node2D


func _input(event):
	if event.is_action_pressed("ui_down"):
		var loaded_net = NeuralNetwork.load($Neural_Net.save_path)
		$Neural_Net.spawn_loaded(loaded_net)
