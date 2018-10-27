from keras import layers, models, optimizers
from keras import backend as K
from keras.initializers import RandomUniform

# Actor model defined using Keras

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, lrate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the learning rate
        self.lrate = lrate
        
        # Build the actor model
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        net = layers.BatchNormalization()(states)
        
        # Add hidden layers
        net = layers.Dense(units=400, activation='relu')(net)
        net = layers.BatchNormalization()(net)
        net = layers.Dense(units=300, activation='relu')(net)
        net = layers.BatchNormalization()(net)

        # Add final output layer with tanh activation
        initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
        actions = layers.Dense(units=self.action_size, activation='tanh',
                               kernel_initializer=initializer, name='actions')(net)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lrate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

# Critic model defined in Keras
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lrate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size

        # Initialize the learning rate
        self.lrate = lrate

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer for state pathway
        net_states = layers.Dense(units=400, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)

        # Combine state and action pathways
        net = layers.Concatenate()([net_states, actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        net = layers.Dense(units=300, activation='relu')(net)
        #net = layers.BatchNormalization()(net)

        # Add final output layer to produce action values (Q values)
        initializer = RandomUniform(minval=-3e-3, maxval=3e-3)
        Q_values = layers.Dense(units=1, kernel_initializer=initializer, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.lrate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)