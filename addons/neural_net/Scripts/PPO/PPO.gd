class_name PPO

var actor: NeuralNetworkAdvanced
var critic: NeuralNetworkAdvanced
var gamma: float = 0.95  # Discount factor for rewards
var epsilon_clip: float = 0.2  # Clipping parameter for PPO
var update_steps: int = 80  # Number of steps to update policy
var max_memory_size: int = 20000  # Maximum size of memory to prevent overflow
var memory: Array = []

func _init(actor_config: Dictionary, critic_config: Dictionary):
	print("Initializing PPO")
	actor = NeuralNetworkAdvanced.new(actor_config)
	critic = NeuralNetworkAdvanced.new(critic_config)
	memory = []

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
	var batch: Array = memory
	memory = []

	for step in range(1):  # Start with a single training step to simplify
		print("Training step: ", step)
		for sample in batch:
			print("Processing sample: ", sample)
			var state: Array = sample["state"]
			var action: int = sample["action"]
			var reward: float = sample["reward"]
			var next_state: Array = sample["next_state"]
			var done: bool = sample["done"]

			# Critic update (value function)
			print("Before critic predict")
			var value: float = critic.predict(state)[0]
			var next_value: float = critic.predict(next_state)[0]
			print("Critic predict done: Value: ", value, " Next Value: ", next_value)

			if check_for_nan([value, next_value]):
				print("Invalid values (NaN) from critic predict!")
				return

			var target: float = reward + (1.0 - float(done)) * gamma * next_value
			var advantage: float = target - value
			print("Critic update: Target: ", target, " Advantage: ", advantage)

			critic.train(state, [target])  # Train the critic

			# Actor update (policy function)
			#print("Before actor predict")
			#print("Actor network weights before training: ", actor.network)  # Print network weights
			var old_probabilities: Array = actor.predict(state)
			var new_probabilities: Array = actor.predict(state)
			#print("Actor predict done")

			if check_for_nan(old_probabilities) or check_for_nan(new_probabilities):
				print("Invalid probabilities (NaN) from actor predict!")
				return

			# Create a one-hot encoded target for the actor
			var action_targets: Array = []
			for i in range(old_probabilities.size()):
				if i == action:
					action_targets.append(advantage)
				else:
					action_targets.append(0.0)
			print("Action targets for training: ", action_targets)

			# Train the actor on the full probability distribution
			#print("Before actor training")
			actor.train(state, action_targets)
			#print("After actor training")
			#print("Actor network weights after training: ", actor.network)

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

func load(path: String):
	print("Loading actor and critic networks.")
	actor.load(path + "_actor")
	critic.load(path + "_critic")
