class_name DDQN
extends DQN

func train_batch(batch: Array):
	for experience in batch:
		# Determine if the batch is time series by checking if the "state" is an array of arrays
		if typeof(experience["state"]) == TYPE_ARRAY and experience["state"].size() > 0 and typeof(experience["state"][0]) == TYPE_ARRAY:
			# Handle time series data
			var sequence_length = experience["state"].size()
			for t in range(sequence_length):
				var current_state = experience["state"][t]
				var next_state = experience["next_state"][t]
				var reward = experience["reward"][t]
				var action = experience["action"][t]
				var done = experience["done"][t]

				# Action selection using the online network
				var next_action = neural_network.predict(next_state).find(neural_network.predict(next_state).max())
				
				# Action evaluation using the target network (or online network if no target network is used)
				var max_future_q: float
				if use_target_network:
					max_future_q = target_neural_network.predict(next_state)[next_action]
				else:
					max_future_q = neural_network.predict(next_state)[next_action]
				
				var target_q_value = reward + discounted_factor * max_future_q if not done else reward
				var target_q_values = neural_network.predict(current_state)
				target_q_values[action] = target_q_value
				neural_network.train(current_state, target_q_values)
		else:
			# Handle regular (non-time series) data
			# Action selection using the online network
			var next_action = neural_network.predict(experience["next_state"]).find(neural_network.predict(experience["next_state"]).max())
			
			# Action evaluation using the target network (or online network if no target network is used)
			var max_future_q: float
			if use_target_network:
				max_future_q = target_neural_network.predict(experience["next_state"])[next_action]
			else:
				max_future_q = neural_network.predict(experience["next_state"])[next_action]
			
			var target_q_value = experience["reward"] + discounted_factor * max_future_q if not experience["done"] else experience["reward"]
			var target_q_values = neural_network.predict(experience["state"])
			target_q_values[experience["action"]] = target_q_value
			neural_network.train(experience["state"], target_q_values)


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
