extends Node2D

var nnas: NeuralNetworkAdvanced

var config = {
	"learning_rate": 0.00001,
	"use_l2_regularization": false,
	"l2_regularization_strength": 0.001,
	"use_adam_optimizer": true,
	"beta1": 0.9,
	"beta2": 0.999,
	"epsilon": 1e-7,
	"early_stopping": false,  # Enable or disable early stopping
	"patience": 100,          # Number of epochs with no improvement after which training will be stopped
	"save_path": "res://earlystoptest.data",  # Path to save the best model
	"smoothing_window": 10,  # Number of epochs to average for loss smoothing
	"check_frequency": 50,    # Frequency of checking early stopping condition
	"minimum_epochs": 1000,   # Minimum epochs before early stopping can trigger
	"improvement_threshold": 0.005,  # Minimum relative improvement required to reset patience
	# Gradient Clipping
	"use_gradient_clipping": true,
	"gradient_clip_value": 1.0,

	# Weight Initialization
	"initialization_type": "he",  # Options are "xavier" or "he"
}

func _ready() -> void:
	nnas = NeuralNetworkAdvanced.new(config)
	nnas.add_layer(3)
	nnas.add_layer(6, nnas.ACTIVATIONS.LEAKY_RELU)
	nnas.add_layer(1, nnas.ACTIVATIONS.LINEAR)
	#nnas.load(nnas.save_path)
	
	#
	for i in range(600):
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
	print(nnas.predict([10.0, 10.0, 10.0]))


func _physics_process(delta: float) -> void:
	if Input.is_action_pressed("predict"):
		for i in range(1):
			# Generate random integer inputs
			var input1 = randi_range(1, 20)
			var input2 = randi_range(1, 20)
			var input3 = randi_range(1, 20)
			
			# The target is the sum of the inputs
			var target = input1 + input2 + input3
			
			# Train the network
			nnas.train([float(input1), float(input2), float(input3)], [float(target)])
		
		print("--------------Prediction--------------")
		print(nnas.predict([1.0, 2.0, 3.0]))
		print(nnas.predict([4.0, 5.0, 6.0]))
		print(nnas.predict([7.0, 8.0, 9.0]))
		print(nnas.predict([10.0, 11.0, 12.0]))
		print(nnas.predict([13.0, 14.0, 15.0]))
		print(nnas.predict([10.0, 10.0, 10.0]))
		$VisualizeNet.visualize(nnas)
