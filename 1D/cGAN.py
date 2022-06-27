from tensorflow.python.keras import Model
import tensorflow as tf
from utils import build_dense


class cLSGAN(Model):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int,
                 critic_layers: list, generator_layers: list,
                 activation='relu', critic_extra_steps=3,
                 critic_dropout=None, generator_dropout=None):
        super(cLSGAN, self).__init__()
        self.d_steps = critic_extra_steps
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.critic = build_dense(input_dim + output_dim, 1, critic_layers,
                                  activation=activation, name='critic',
                                  dropout=critic_dropout)
        self.generator = build_dense(input_dim + latent_dim, output_dim, generator_layers,
                                     activation=activation, name='generator',
                                     dropout=generator_dropout)

    def compile(self, d_optimizer, g_optimizer):
        """Compile the model by setting the optimizer for both critic ond generator"""
        super(cLSGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_critic(self, X, y):
        """Training step for the critic model"""
        batch_size = tf.shape(X)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)
        generator_input = tf.concat([X, random_latent_vectors], axis=1)
        critic_real_input = tf.concat([X, y], axis=1)

        with tf.GradientTape() as tape:
            fake = self.generator(generator_input, training=True)
            critic_fake_input = tf.concat([X, fake], axis=1)

            fake_logits = self.critic(critic_fake_input, training=True)
            real_logits = self.critic(critic_real_input, training=True)

            d_loss = self.critic_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

        d_gradient = tape.gradient(d_loss, self.critic.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradient, self.critic.trainable_variables))
        return d_loss

    def train_generator(self, X, ):
        """Training step for the generator model"""
        batch_size = tf.shape(X)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)
        input_vector = tf.concat([X, random_latent_vectors], axis=1)

        with tf.GradientTape() as tape:
            fake = self.generator(input_vector, training=True)
            critic_fake_input = tf.concat([X, fake], axis=1)
            # Get the discriminator logits for fake images
            fake_logits = self.critic(critic_fake_input, training=True)
            # Calculate the generator loss
            g_loss = self.generator_loss_fn(fake_logits)
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))
        return g_loss

    def train_step(self, data):
        """Train iteratively both the critic and the generator"""
        # Train the critic
        X, y = data
        for i in range(self.d_steps):
            d_loss = self.train_critic(X, y)

        # Train the generator
        g_loss = self.train_generator(X)
        return {"d_loss": d_loss, "g_loss": g_loss}

    def critic_loss_fn(self, real_logits, fake_logits):
        """Compute the critic loss. It expects real and fake logits as real numbers"""
        real_loss = tf.reduce_mean(tf.square(real_logits - tf.ones_like(real_logits)))
        fake_loss = tf.reduce_mean(tf.square(fake_logits + tf.ones_like(fake_logits)))
        return fake_loss + real_loss

    def generator_loss_fn(self, fake_logits):
        """Compute the generator loss. It expects real and fake logits as real numbers"""
        return tf.reduce_mean(tf.square(fake_logits))
