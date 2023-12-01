import tensorflow as tf

from keras import Model
from keras import layers

from keras.src.engine.keras_tensor import KerasTensor


def PolicyNetwork(
    state_shape: tuple[int, ...], 
    action_bounds: list[tuple[float, float]], 
    units: int = 256, 
    activation: str = 'relu',
    batch_norm: bool = False, 
    dropout: float = 0.0
) -> Model:
    
    state_input = layers.Input(shape=state_shape, name='state_input')

    x_shared = hidden_layer(state_input, units, activation, batch_norm, dropout)
    x_shared = hidden_layer(x_shared, units, activation, batch_norm, dropout)

    x_actions = []
    for bounds in action_bounds:
        x = hidden_layer(x_shared, units, activation, batch_norm, dropout)
        x = policy_output_layer(x, bounds)
        x_actions.append(x)

    outputs = layers.Concatenate()(x_actions)

    return Model(state_input, outputs)


def ValueNetwork(
    state_shape: tuple[int, ...], 
    action_shape: tuple[int, ...], 
    units: int = 256, 
    activation: str = 'relu',
    batch_norm: bool = False,
    dropout: float = 0.0
) -> Model:
    
    state_input = layers.Input(shape=state_shape, name='state_input')
    action_input = layers.Input(shape=action_shape, name='action_input')

    x_state  = hidden_layer(state_input, units, activation, batch_norm, dropout)
    x_state  = hidden_layer(x_state, units, activation, batch_norm, dropout)

    x_action = hidden_layer(action_input, units, activation, batch_norm, dropout) 
    x_action = hidden_layer(x_action, units, activation, batch_norm, dropout)

    x = layers.Concatenate()([x_state, x_action])

    x = hidden_layer(x, units, activation, batch_norm, dropout)
    x = hidden_layer(x, units, activation, batch_norm, dropout)

    outputs = value_output_layer(x, units=1)

    return Model([state_input, action_input], outputs)


def BoundedSigmoid(minval: float, maxval: float) -> callable:
    def bounded_sigmoid(x):
        return (maxval-minval) * tf.nn.sigmoid(x) + minval
    return bounded_sigmoid


def hidden_layer(
    x: KerasTensor, 
    units: int, 
    activation: str, 
    batch_norm: bool, 
    dropout: float
) -> KerasTensor:
    x = layers.Dense(units)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x

def policy_output_layer(
    x: KerasTensor, 
    bounds: tuple[float, float]
) -> KerasTensor:
    'Produces actions.'
    return layers.Dense(1, BoundedSigmoid(*bounds))(x)

def value_output_layer(
    x: KerasTensor, 
    units: int
) -> KerasTensor:
    'Produces Q-values associated with the (state, action) pairs.'
    return layers.Dense(units)(x)

