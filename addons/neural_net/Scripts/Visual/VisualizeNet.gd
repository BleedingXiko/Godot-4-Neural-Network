extends Node2D

@export var sprite_texture: Texture
@export var display_bias: bool = false
@export var y_separation: float = 100.0
@export var x_separation: float = 200.0
@export var positive_color: Color = Color.BLUE
@export var negative_color: Color = Color.RED
@export var zero_color: Color = Color.BLACK
@export var min_color_threshold: float = -1.0 # The value inside the matrix at which we color the line 100% negative
@export var max_color_threshold: float = 1.0  # The value inside the matrix at which we color the line 100% positive
@export var min_line_width: float = 3.0
@export var max_line_width: float = 12.0
@export var min_width_threshold: float = 0.0  # The value inside the matrix at which the line will be of min_line_width px wide
@export var max_width_threshold: float = 3.0  # The value inside the matrix at which the line will be of max_line_width px wide
@export var dead_weight_threshold: float = -0.005
@export var dead_weight_line_width: float = 1.0

func visualize(nn: NeuralNetworkAdvanced):
	clear()

	var anchor = Vector2(0, 0)
	var head_pts: Array[Vector2] = []
	var tail_pts: Array[Vector2] = []

	var max_node_height = _get_max_node_height(nn)

	# Visualize input layer nodes
	for j in range(nn.layer_structure[0]):
		var d = Sprite2D.new()
		d.texture = sprite_texture
		var y_margin = (max_node_height - nn.layer_structure[0]) * y_separation / 2.0
		d.position.y = j * y_separation + anchor.y + y_margin
		d.position.x = anchor.x
		add_child(d)
		d.show()
		head_pts.append(d.position)

	# Visualize hidden layers and output layer nodes
	for i in range(nn.network.size()):
		var layer = nn.network[i]

		var x_offset = (i + 1) * x_separation + anchor.x
		var weights = layer["weights"].get_data()  # Use correct method to get matrix data
		var node_count = layer["weights"].get_rows()  # Updated to use get_rows for node count

		var y_margin = (max_node_height - node_count) * y_separation / 2.0
		for j in range(node_count):
			var d = Sprite2D.new()  # Create a new instance each time
			d.texture = sprite_texture
			if display_bias:
				var valB = layer["bias"].get_at(j, 0)  # Access the bias value using get_at
				var color_ratioB = remap(valB, min_color_threshold, max_color_threshold, 0.0, 1.0)
				d.modulate = lerp(Color.BLACK, Color.WHITE, color_ratioB)
				d.name = str(valB)
			d.position = Vector2(x_offset, j * y_separation + anchor.y + y_margin)
			add_child(d)
			d.show()
			tail_pts.append(d.position)

			# Visualize connections (weights) between nodes
			for h in range(head_pts.size()):
				for t in range(tail_pts.size()):
					var val = layer["weights"].get_at(t, h)  # Access the weight value using get_at
					var color_ratio = remap(val, min_color_threshold, max_color_threshold, 0.0, 1.0)
					var l = Line2D.new()  # Create a new instance of the line each time
					l.points = [head_pts[h], tail_pts[t]]
					if abs(val) < dead_weight_threshold:
						l.default_color = zero_color
						l.width = dead_weight_line_width
					else:
						var width = abs(val)
						width = remap(width, min_width_threshold, max_width_threshold, min_line_width, max_line_width)
						l.width = clamp(width, min_line_width, max_line_width)
						l.default_color = lerp(negative_color, positive_color, color_ratio)

					l.show_behind_parent = true
					l.name = "Line" + str(i) + "-" + str(j)

					add_child(l)
					l.show()

		head_pts = tail_pts.duplicate(true)
		tail_pts.clear()

func _get_max_node_height(nn: NeuralNetworkAdvanced) -> int:
	var max_node_count: int = 0
	for layer in nn.network:
		var layer_size: int = layer["weights"].get_rows()  # Updated to use get_rows for row count
		max_node_count = max(layer_size, max_node_count)

	return max_node_count

func clear():
	# Clear existing child nodes before adding new ones
	for n in get_children():
		n.queue_free()
