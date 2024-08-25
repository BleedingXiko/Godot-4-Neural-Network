class_name PPO

var actor: NeuralNetworkAdvanced
var critic: NeuralNetworkAdvanced
var target_critic: NeuralNetworkAdvanced

# Hyperparameters
var gamma: float = 0.95  # Discount factor for rewards
var epsilon_clip: float = 0.2  # Clipping parameter for PPO
var update_steps: int = 80  # Number of steps to update policy
var max_memory_size: int = 10000  # Maximum size of memory to prevent overflow
var memory: Array = []
var batch_size: int = 64  # Size of mini-batch
var lambda: float = 0.95  # GAE parameter, optional
var entropy_beta: float = 0.01  # Entropy coefficient, optional
var initial_learning_rate: float = 0.001  # Initial learning rate
var min_learning_rate: float = 0.0001  # Minimum learning rate
var decay_rate: float = 0.99  # Decay rate for exponential decay
var clip_value: float = 0.2  # Gradient clipping value, optional
var target_network_update_steps: int = 1000  # Steps to update target network
var learning_rate_schedule_type: String = "constant"  # Type of learning rate scheduling
var accuracy_threshold: float = 0.95  # Threshold for accuracy-based learning rate adjustment

# Optional flags to enable or disable features
var use_gae: bool = true
var use_entropy: bool = true
var use_target_network: bool = true
var use_gradient_clipping: bool = true
var use_learning_rate_scheduling: bool = true

# For accuracy-based learning rate adjustment
var previous_accuracy: float = 0.0
var current_accuracy: float = 0.0

func _init(actor_config: Dictionary, critic_config: Dictionary):
	#print("Initializing PPO")
	actor = NeuralNetworkAdvanced.new(actor_config)
	critic = NeuralNetworkAdvanced.new(critic_config)
	set_config({})
	if use_target_network:
		target_critic = NeuralNetworkAdvanced.new(critic_config)
		update_target_network()  # Initialize target network with critic's weights
	memory = []

func set_config(config: Dictionary) -> void:
	gamma = config.get("gamma", gamma)
	epsilon_clip = config.get("epsilon_clip", epsilon_clip)
	update_steps = config.get("update_steps", update_steps)
	max_memory_size = config.get("max_memory_size", max_memory_size)
	batch_size = config.get("batch_size", batch_size)
	lambda = config.get("lambda", lambda)
	entropy_beta = config.get("entropy_beta", entropy_beta)
	initial_learning_rate = config.get("initial_learning_rate", initial_learning_rate)
	min_learning_rate = config.get("min_learning_rate", min_learning_rate)
	decay_rate = config.get("decay_rate", decay_rate)
	clip_value = config.get("clip_value", clip_value)
	target_network_update_steps = config.get("target_network_update_steps", target_network_update_steps)
	learning_rate_schedule_type = config.get("learning_rate_schedule_type", learning_rate_schedule_type)
	accuracy_threshold = config.get("accuracy_threshold", accuracy_threshold)
	use_gae = config.get("use_gae", use_gae)
	use_entropy = config.get("use_entropy", use_entropy)
	use_target_network = config.get("use_target_network", use_target_network)
	use_gradient_clipping = config.get("use_gradient_clipping", use_gradient_clipping)
	use_learning_rate_scheduling = config.get("use_learning_rate_scheduling", use_learning_rate_scheduling)
	

func softmax(output_array: Array) -> Array:
	var max_val: float = output_array.max()
	var exp_values: Array = []
	var sum_exp: float = 0.0

	for value in output_array:
		var exp_val: float = exp(value - max_val)
		exp_values.append(exp_val)
		sum_exp += exp_val

	var softmax_output: Array = []
	for exp_val in exp_values:
		softmax_output.append(exp_val / sum_exp)

	return softmax_output

func get_action(state: Array) -> int:
	#print("Getting action for state: ", state)
	var raw_output: Array = actor.predict(state)  # Get the raw output from the actor network
	#print("Raw output from actor: ", raw_output)
	if check_for_nan(raw_output):
		#print("NaN detected in raw output!")
		return 0  # Return a default action to avoid crash

	var probabilities: Array = softmax(raw_output)  # Apply softmax to convert to probabilities
	#print("Probabilities after softmax: ", probabilities)
	if check_for_nan(probabilities):
		print("NaN detected in probabilities!")
		return 0  # Return a default action to avoid crash

	return sample_action(probabilities)

func sample_action(probabilities: Array) -> int:
	var r: float = randf()
	var cumulative_probability: float = 0.0
	for i in range(probabilities.size()):
		cumulative_probability += probabilities[i]
		if r < cumulative_probability:
			#print("Sampled action: ", i)
			return i
	return probabilities.size() - 1

func keep(state: Array, action: int, reward: float, next_state: Array, done: bool):
	#print("Storing experience: State: ", state, " Action: ", action, " Reward: ", reward, " Next State: ", next_state, " Done: ", done)
	if memory.size() >= max_memory_size:
		memory.pop_front()  # Remove the oldest experience if memory exceeds limit
	memory.append({
		"state": state,
		"action": action,
		"reward": reward,
		"next_state": next_state,
		"done": done
	})
	#print("Experience stored. Current memory size: ", memory.size())

func train():
	if memory.size() == 0:
		print("No memory to train on.")
		return

	var num_batches = memory.size() / batch_size
	if num_batches == 0:
		num_batches = 1  # Ensure at least one batch is processed

	for step in range(update_steps):  # Adjust update steps as necessary
		# Shuffle memory before creating mini-batches
		memory.shuffle()

		for batch_idx in range(num_batches):
			var start_idx = batch_idx * batch_size
			var end_idx = min(start_idx + batch_size, memory.size())
			var mini_batch = memory.slice(start_idx, end_idx, true)

			for sample in mini_batch:
				var state: Array = sample["state"]
				var action: int = sample["action"]
				var reward: float = sample["reward"]
				var next_state: Array = sample["next_state"]
				var done: bool = sample["done"]

				# Critic update (value function)
				var value: float = critic.predict(state)[0]

				var next_value: float
				if use_target_network:
					next_value = target_critic.predict(next_state)[0]
				else:
					next_value = critic.predict(next_state)[0]

				if check_for_nan([value, next_value]):
					print("Invalid values (NaN) from critic predict!")
					return

				var target: float = reward + (1.0 - float(done)) * gamma * next_value
				var advantage: float = target - value
				critic.train(state, [target])  # Train the critic

				# Actor update (policy function)
				var old_probabilities: Array = actor.predict(state)
				if check_for_nan(old_probabilities):
					print("Invalid probabilities (NaN) from actor predict!")
					return

				# Create a one-hot encoded target for the actor with advantage
				var action_targets: Array = []
				for i in range(old_probabilities.size()):
					if i == action:
						action_targets.append(advantage)
					else:
						action_targets.append(0.0)

				# Calculate the entropy of the action distribution
				if use_entropy:
					var entropy: float = compute_entropy(old_probabilities)
					var loss_with_entropy: Array = []
					for j in range(action_targets.size()):
						loss_with_entropy.append(action_targets[j] - entropy_beta * entropy)
					action_targets = loss_with_entropy

				if use_gradient_clipping:
					# Train the actor on the full probability distribution with gradient clipping
					actor.train(state, clip_gradients(action_targets))
				else:
					# Train the actor on the full probability distribution without gradient clipping
					actor.train(state, action_targets)

			if use_learning_rate_scheduling:
				# Optionally update learning rate
				update_learning_rate(step)

			# Update target network after every `target_network_update_steps`
			if use_target_network and step % target_network_update_steps == 0:
				update_target_network()

		# Compute accuracy or other performance metrics
		current_accuracy = compute_accuracy()

		# Adjust learning rate based on accuracy improvement
		if use_learning_rate_scheduling and learning_rate_schedule_type == "accuracy_based":
			adjust_learning_rate_based_on_accuracy()

	#print("Training step completed.")


func compute_accuracy() -> float:
	# Dummy function to compute accuracy. Replace with actual computation.
	var correct_predictions: int = 0
	var total_predictions: int = memory.size()
	for sample in memory:
		var state: Array = sample["state"]
		var action: int = sample["action"]
		var predicted_action: int = get_action(state)
		if predicted_action == action:
			correct_predictions += 1
	return float(correct_predictions) / float(total_predictions)

func adjust_learning_rate_based_on_accuracy():
	if current_accuracy < accuracy_threshold:
		if current_accuracy > previous_accuracy:
			# If accuracy is improving but below the threshold, reduce the learning rate slightly
			actor.learning_rate = max(min_learning_rate, actor.learning_rate * decay_rate)
			critic.learning_rate = max(min_learning_rate, critic.learning_rate * decay_rate)
			if use_target_network:
				target_critic.learning_rate = max(min_learning_rate, target_critic.learning_rate * decay_rate)
		elif current_accuracy < previous_accuracy:
			# If accuracy is worsening, reduce the learning rate more aggressively
			actor.learning_rate = max(min_learning_rate, actor.learning_rate * (decay_rate ** 2))
			critic.learning_rate = max(min_learning_rate, critic.learning_rate * (decay_rate ** 2))
			if use_target_network:
				target_critic.learning_rate = max(min_learning_rate, target_critic.learning_rate * (decay_rate ** 2))
	else:
		# If accuracy meets or exceeds the threshold, slowly increase the learning rate to encourage further improvement
		actor.learning_rate = min(initial_learning_rate, actor.learning_rate / decay_rate)
		critic.learning_rate = min(initial_learning_rate, critic.learning_rate / decay_rate)
		if use_target_network:
			target_critic.learning_rate = min(initial_learning_rate, target_critic.learning_rate / decay_rate)

	# Update previous accuracy for the next iteration
	previous_accuracy = current_accuracy


func check_for_nan(array: Array) -> bool:
	for value in array:
		if is_nan(value):
			return true
	return false

func is_nan(value: float) -> bool:
	return value != value  # NaN is the only value that is not equal to itself

func save(path: String):
	print("Saving actor and critic networks.")
	actor.save(path + "_actor")
	critic.save(path + "_critic")

func load(path: String):
	print("Loading actor and critic networks.")
	actor.load(path + "_actor")
	critic.load(path + "_critic")
	if use_target_network:
		update_target_network()

func compute_entropy(probabilities: Array) -> float:
	var entropy: float = 0.0
	for prob in probabilities:
		entropy -= prob * log(prob + 1e-10)
	return entropy

func update_target_network():
	target_critic = critic.copy()

func clip_gradients(gradients: Array) -> Array:
	for i in range(gradients.size()):
		gradients[i] = clamp(gradients[i], -clip_value, clip_value)
	return gradients

func update_learning_rate(step: int):
	var new_lr: float
	if learning_rate_schedule_type == "exponential_decay":
		new_lr = max(min_learning_rate, initial_learning_rate * pow(decay_rate, step))
	elif learning_rate_schedule_type == "linear_decay":
		new_lr = max(min_learning_rate, initial_learning_rate - (initial_learning_rate - min_learning_rate) * (float(step) / float(update_steps)))
	elif learning_rate_schedule_type == "step_decay":
		var decay_factor = 0.5  # Example decay factor
		var decay_steps = 100  # Example step interval
		new_lr = max(min_learning_rate, initial_learning_rate * pow(decay_factor, float(step) / float(decay_steps)))
	else:  # Constant or no scheduling
		new_lr = initial_learning_rate
	actor.learning_rate = new_lr
	critic.learning_rate = new_lr
	if use_target_network:
		target_critic.learning_rate = new_lr
