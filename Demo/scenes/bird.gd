extends CharacterBody2D

const GRAVITY : int = 2000
const MAX_VEL : int = 300
var vel: Vector2 = Vector2()
const FLAP_SPEED : int = -300
const START_POS = Vector2(100, 400)

var nn: NeuralNetwork

var score = 0


func think(pipe_pos):
	var inputs = []
		#inputs.append(global_position.x)
#	inputs.append(global_position.y)
#	inputs.append(pipe_pos.x)
#	inputs.append(pipe_pos.y)
	inputs.append(global_position.y)
	#inputs.append(motion.y / 1000)
	inputs.append(pipe_pos.x)
	inputs.append(pipe_pos.y)
	
	var outputs = nn.predict(inputs)
	if outputs[0] > outputs[1]:
		flap()
	#dist_to_pipey =  (global_position.y / pipe_pos.y)

# Called when the node enters the scene tree for the first time.
func _physics_process(delta):
	var pipe_pos = get_parent().get_parent().pipes[0].global_position
	think(pipe_pos)
	vel.y += GRAVITY * delta
	#terminal velocity
	if vel.y > MAX_VEL:
		vel.y = MAX_VEL
		rotation_degrees += 4
		$AnimatedSprite2D.play()
		
	var col = move_and_collide(vel)
	if col:
		$AnimatedSprite2D.stop()
		queue_free()
		
func flap():
	rotation_degrees = -30
	velocity.y = FLAP_SPEED
