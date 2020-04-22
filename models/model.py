import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Concatenate, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx('float64')

# Actor model defined using Keras

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, name="Actor"):
        """Initialize parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Dimensions of 1st hidden layer
            fc2_units (int): Dimensions of 2nd hidden layer
            name (string): Name of the model
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.name = name
        
        # Build the actor model
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = Input(shape=(self.state_size,))
        
        # Add hidden layers
        net = Dense(units=self.fc1_units, activation='relu', kernel_initializer='glorot_uniform')(states)

        # Add final output layer with tanh activation
        actions = Dense(units=self.action_size, activation='tanh', kernel_initializer='glorot_uniform')(net)

        # Create Keras model
        self.model = Model(inputs=states, outputs=actions, name=self.name)

# Critic model defined in Keras
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=256, fc2_units=256, fc3_units=128, name="Critic"):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Dimensions of 1st hidden layer
            fc2_units (int): Dimensions of 2nd hidden layer
            name (string): Name of the model
        """

        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.fc3_units = fc3_units
        self.name = name

        # Build the critic model
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = Input(shape=(self.state_size,))
        actions = Input(shape=(self.action_size,))

        # Add hidden layer for state pathway
        net_states = Dense(units=self.fc1_units, kernel_initializer='glorot_uniform')(states)
        net_states = LeakyReLU()(net_states)

        # Combine state and action pathways
        net = Concatenate(axis=-1)([net_states, actions])

        # Add more layers to the combined network
        net = Dense(units=self.fc2_units, kernel_initializer='glorot_uniform')(net)
        net = LeakyReLU()(net)
        net = Dense(units=self.fc3_units, kernel_initializer='glorot_uniform')(net)
        net = LeakyReLU()(net)
        
        # Add final output layer to produce action values (Q values)
        Q_values = Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0001))(net)

        # Create Keras model
        self.model = Model(inputs=[states, actions], outputs=Q_values, name=self.name)

