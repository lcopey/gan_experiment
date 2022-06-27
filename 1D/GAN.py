import tensorflow as tf
from tensorflow.python.keras import Model
from utils import build_dense


class WGANGP(Model):
    def __init__(self, input_dim: int, latent_dim: int,
                 critic_layers: list, generator_layers: list,
                 activation='relu', critic_extra_steps=3,
                 gp_weight=10.,
                 critic_dropout=None, generator_dropout=None):
        super(WGANGP, self).__init__()
        self.d_steps = critic_extra_steps
        self.gp_weight = gp_weight
        self.latent_dim = latent_dim
        self.critic = build_dense(input_dim, 1, critic_layers,
                                  activation=activation, name='critic',
                                  dropout=critic_dropout)
        self.generator = build_dense(latent_dim, input_dim, generator_layers,
                                     activation=activation, name='generator',
                                     dropout=generator_dropout)

    def compile(self, d_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, X, fake):
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * fake + X * (1 - alpha)
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.reduce_sum(tf.square(grads), axis=1)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_critic(self, batch_size, X):
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)
        with tf.GradientTape() as tape:
            fake = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(fake, training=True)
            real_logits = self.critic(X, training=True)

            d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

            gp = self.gradient_penalty(batch_size, X, fake)

            d_loss = d_cost + gp * self.gp_weight

        d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))
        return d_loss

    def train_generator(self, batch_size):
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), )
        with tf.GradientTape() as tape:
            fake = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            fake_logits = self.critic(fake, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(fake_logits)
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        return g_loss

    def train_step(self, X):
        batch_size = tf.shape(X)[0]

        # Train the critic
        for i in range(self.d_steps):
            d_loss = self.train_critic(batch_size, X)

        # Train the generator
        g_loss = self.train_generator(batch_size)
        return {"d_loss": d_loss, "g_loss": g_loss}

    def d_loss_fn(self, real_logits, fake_logits):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def g_loss_fn(self, fake_logits):
        return -tf.reduce_mean(fake_logits)


class LSGAN(Model):
    def __init__(self, output_dim: int, latent_dim: int,
                 critic_layers: list, generator_layers: list,
                 activation='relu', critic_extra_steps=3,
                 critic_dropout=None, generator_dropout=None):
        super(LSGAN, self).__init__()
        self.d_steps = critic_extra_steps
        self.latent_dim = latent_dim
        self.critic = build_dense(output_dim, 1, critic_layers,
                                  activation=activation, name='critic',
                                  dropout=critic_dropout)
        self.generator = build_dense(latent_dim, output_dim, generator_layers,
                                     activation=activation, name='generator',
                                     dropout=generator_dropout)

    def compile(self, d_optimizer, g_optimizer):
        """Compile the model by setting the optimizer for both critic ond generator"""
        super(LSGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_critic(self, batch_size, X):
        """Training step for the critic model"""
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)
        with tf.GradientTape() as tape:
            fake = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(fake, training=True)
            real_logits = self.critic(X, training=True)

            d_loss = self.critic_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

        d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))
        return d_loss

    def train_generator(self, batch_size):
        """Training step for the generator model"""
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), )
        with tf.GradientTape() as tape:
            fake = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            fake_logits = self.critic(fake, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss_fn(fake_logits)
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        return g_loss

    def train_step(self, X):
        """Train iteratively both the critic and the generator"""
        batch_size = tf.shape(X)[0]

        # Train the critic
        for i in range(self.d_steps):
            d_loss = self.train_critic(batch_size, X)

        # Train the generator
        g_loss = self.train_generator(batch_size)
        return {"d_loss": d_loss, "g_loss": g_loss}

    def critic_loss_fn(self, real_logits, fake_logits):
        """Compute the critic loss. It expects real and fake logits as real numbers"""
        real_loss = tf.reduce_mean(tf.square(real_logits - tf.ones_like(real_logits)))
        fake_loss = tf.reduce_mean(tf.square(fake_logits + tf.ones_like(fake_logits)))
        return fake_loss + real_loss

    def generator_loss_fn(self, fake_logits):
        """Compute the generator loss. It expects real and fake logits as real numbers"""
        return tf.reduce_mean(tf.square(fake_logits))
