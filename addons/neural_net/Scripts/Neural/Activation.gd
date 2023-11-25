class_name Activation


var functions: Dictionary = {
	"SIGMOID": {
		"function": Callable(Activation, "sigmoid"),
		"derivative": Callable(Activation, "dsigmoid"),
		"name": "SIGMOID",
	},
	"RELU": {
		"function": Callable(Activation, "relu"),
		"derivative": Callable(Activation, "drelu"),
		"name": "RELU"
	},
	"TANH": {
		"function": Callable(Activation, "tanh_"),
		"derivative": Callable(Activation, "dtanh"),
		"name": "TANH"
	},
	"ARCTAN": {
		"function": Callable(Activation, "arcTan"),
		"derivative": Callable(Activation, "darcTan"),
		"name": "ARCTAN"
	},
	"PRELU": {
		"function": Callable(Activation, "prelu"),
		"derivative": Callable(Activation, "dprelu"),
		"name": "PRELU"
	},
	"ELU": {
		"function": Callable(Activation, "elu"),
		"derivative": Callable(Activation, "delu"),
		"name": "ELU"
	},
	"SOFTPLUS": {
		"function": Callable(Activation, "softplus"),
		"derivative": Callable(Activation, "dsoftplus"),
		"name": "SOFTPLUS"
	},
	"SWISH": {
		"function": Callable(Activation, "swish"),
		"derivative": Callable(Activation, "dswish"),
		"name": "SWISH"
	},
	"MISH": {
		"function": Callable(Activation, "mish"),
		"derivative": Callable(Activation, "dmish"),
		"name": "MISH"
	},
	 "LINEAR": {
		"function": Callable(Activation, "linear"),
		"derivative": Callable(Activation, "dlinear"),
		"name": "LINEAR"
	}
}

static func sigmoid(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + exp(-value))

static func dsigmoid(value: float, _row: int, _col: int) -> float:
	return value * (1 - value)

static func relu(value: float, _row: int, _col: int) -> float:
	return max(0.0, value)

static func drelu(value: float, _row: int, _col: int) -> float:
	if value < 0:
		return 0.0
	else:
		return 1.0

static func tanh_(value: float, _row: int, _col: int) -> float:
	return tanh(value)

static func dtanh(value: float, _row: int, _col: int) -> float:
	return 1 - pow(tanh(value), 2)

static func arcTan(value: float, _row: int, _col: int) -> float:
	return atan(value)

static func darcTan(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + pow(value, 2))

static func prelu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	return (alpha * value) if value < 0 else value

static func dprelu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	return alpha if value < 0 else 1

static func elu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	if value < 0:
		return alpha * (exp(value) - 1)
	else:
		return value

static func delu(value: float, _row: int, _col: int) -> float:
	var alpha: float = 0.1
	if value < 0:
		return alpha * exp(value)
	else:
		return 1.0

static func softplus(value: float, _row: int, _col: int) -> float:
	return log(1 + exp(value))

static func dsoftplus(value: float, _row: int, _col: int) -> float:
	return 1 / (1 + exp(-value))

static func swish(value: float, _row: int, _col: int, beta: float = 1.0) -> float:
	return value * sigmoid(beta * value, _row, _col)

static func dswish(value: float, _row: int, _col: int, beta: float = 1.0) -> float:
	return beta * value * dsigmoid(beta * value, _row, _col) + sigmoid(beta * value, _row, _col)

static func mish(value: float, _row: int, _col: int) -> float:
	return value * tanh_(softplus(value, _row, _col), _row, _col)

static func dmish(value: float, _row: int, _col: int) -> float:
	var sp = softplus(value, _row, _col)
	var tsp = tanh_(sp, _row, _col)
	return tsp + value * dtanh(sp, _row, _col) * dsoftplus(value, _row, _col)

static func linear(value: float, _row: int, _col: int) -> float:
	return value

static func dlinear(value: float, _row: int, _col: int) -> float:
	return 1.0

