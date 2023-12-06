extends TouchScreenButton


var data = 0

func _on_cell_pressed():
	match data:
		0:
			data = 1
		1:
			data = 0




func _process(delta):
	match data:
		0:
			modulate = Color.BLACK
		1:
			modulate = Color.WHITE
