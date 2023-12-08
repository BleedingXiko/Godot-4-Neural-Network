extends Node2D

func _ready():
	var maps = ["res://Demo/NN/track.tscn", "res://Demo/NN/track_2.tscn"]
	var picked = maps.pick_random()
	var scene = load(picked)
	var child = scene.instantiate()
	add_child(child)

func _input(event):
	if event.is_action_pressed("ui_down"):
		$Neural_Net.spawn_loaded()
