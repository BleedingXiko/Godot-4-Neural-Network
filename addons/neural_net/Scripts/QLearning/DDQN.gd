class_name DDQN
extends DQN

func _init(config: Dictionary) -> void:
	super(config)  # Call the base class's init method
	# Ensure the target network is initialized properly
	if use_target_network:
		target_neural_network = neural_network.copy()

func train(current_states: Array, reward_of_previous_state: float, done: bool = false) -> int:
	var current_q_values = neural_network.predict(current_states)
	var current_action = choose_action(current_states)
	
	# Handle the learning and updating process
	if previous_state.size() != 0:
		if use_replay:
			# Using replay memory
			add_to_memory(previous_state, previous_action, reward_of_previous_state, current_states, done)
			if replay_memory.size() >= batch_size:
				var batch = sample_replay_memory(sampling_strategy)
				train_batch(batch)
		else:
			# Immediate Q-value update without replay
			var next_action = neural_network.predict(current_states).find(current_q_values.max())
			var max_future_q: float
			if use_target_network:
				max_future_q = target_neural_network.predict(current_states)[next_action]
			else:
				max_future_q = current_q_values[next_action]

			var target_q_value = reward_of_previous_state + discounted_factor * max_future_q if not done else reward_of_previous_state
			var target_q_values = neural_network.predict(previous_state)
			target_q_values[previous_action] = target_q_value
			neural_network.train(previous_state, target_q_values)

	# Update previous state and action for the next step
	previous_state = current_states
	previous_action = current_action
	
	# Handle target network updates
	if steps_completed % update_target_every_steps == 0:
		update_target_network()

	if steps_completed % decay_per_steps == 0:
		if print_debug_info:
			print("Total steps completed:", steps_completed)
			print("Current exploration probability:", exploration_probability)
			print("Q-Net data:", neural_network.debug())
			print("-----------------------------------------------------------------------------------------")
	
	steps_completed += 1
	return previous_action
