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
var initial_learning_rate: float = 0.001  # Optional learning rate scheduling
var min_learning_rate: float = 0.0001  # Optional learning rate scheduling
var decay_rate: float = 0.99  # Optional learning rate scheduling
var clip_value: float = 0.2  # Gradient clipping value, optional

# Optional flags to enable or disable features
var use_gae: bool = true
var use_entropy: bool = true
var use_target_network: bool = true
var use_gradient_clipping: bool = true
var use_learning_rate_scheduling: bool = true

func _init(actor_config: Dictionary, critic_config: Dictionary):
	print("Initializing PPO")
	actor = NeuralNetworkAdvanced.new(actor_config)
	critic = NeuralNetworkAdvanced.new(critic_config)
	set_config({})
	if use_target_network:
		target_critic = NeuralNetworkAdvanced.new(critic_config)
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
	print("Getting action for state: ", state)
	var raw_output: Array = actor.predict(state)  # Get the raw output from the actor network
	print("Raw output from actor: ", raw_output)
	if check_for_nan(raw_output):
		print("NaN detected in raw output!")
		return 0  # Return a default action to avoid crash

	var probabilities: Array = softmax(raw_output)  # Apply softmax to convert to probabilities
	print("Probabilities after softmax: ", probabilities)
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
			print("Sampled action: ", i)
			return i
	return probabilities.size() - 1

func remember(state: Array, action: int, reward: float, next_state: Array, done: bool):
	print("Storing experience: State: ", state, " Action: ", action, " Reward: ", reward, " Next State: ", next_state, " Done: ", done)
	if memory.size() >= max_memory_size:
		memory.pop_front()  # Remove the oldest experience if memory exceeds limit
	memory.append({
		"state": state,
		"action": action,
		"reward": reward,
		"next_state": next_state,
		"done": done
	})
	print("Experience stored. Current memory size: ", memory.size())

func train():
	if memory.size() == 0:
		print("No memory to train on.")
		return

	print("Training PPO with memory size: ", memory.size())

	for step in range(update_steps):  # Adjust update steps as necessary
		print("Training step: ", step)
		for sample in memory:
			print("Processing sample: ", sample)
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

			if use_entropy:
				# Calculate the entropy of the action distribution
				var entropy: float = compute_entropy(old_probabilities)
				# Add entropy regularization to the advantage
				advantage += entropy_beta * entropy

			# Create a one-hot encoded target for the actor with advantage and entropy
			var action_targets: Array = []
			for i in range(old_probabilities.size()):
				if i == action:
					action_targets.append(advantage)
				else:
					action_targets.append(0.0)

			if use_gradient_clipping:
				# Train the actor on the full probability distribution with gradient clipping
				actor.train(state, clip_gradients(action_targets))
			else:
				# Train the actor on the full probability distribution without gradient clipping
				actor.train(state, action_targets)

		if use_learning_rate_scheduling:
			# Optionally update learning rate
			update_learning_rate(step)

	if use_target_network:
		# Update target network
		update_target_network()

	print("Training step completed.")

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
	if use_target_network:
		target_critic.save(path + "_target_critic")

func load(path: String):
	print("Loading actor and critic networks.")
	actor.load(path + "_actor")
	critic.load(path + "_critic")
	if use_target_network:
		target_critic.load(path + "_target_critic")

func compute_gae(rewards: Array, values: Array, dones: Array) -> Array:
	var advantages: Array = []
	var gae: float = 0.0
	for i in range(rewards.size() - 1, -1, -1):
		var delta: float = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
		gae = delta + gamma * lambda * (1 - dones[i]) * gae
		advantages.insert(0, gae)
	return advantages

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
	var new_lr: float = max(min_learning_rate, initial_learning_rate * pow(decay_rate, step))
	actor.learning_rate = new_lr
	critic.learning_rate = new_lr
	if use_target_network:
		target_critic.learning_rate = new_lr
