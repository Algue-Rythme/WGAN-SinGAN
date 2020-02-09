import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class Generator(tf.keras.models.Model):

    def __init__(self):
        super(Generator, self).__init__()
        lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
        self.conv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), dilation_rate=2, activation=lrelu)
        self.conv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), activation=lrelu)
        self.conv3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), dilation_rate=2, activation=lrelu)
        self.conv4 = Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=lrelu)
        self.conv5 = Conv2DTranspose(filters=16, kernel_size=(5, 5), dilation_rate=2, activation=lrelu)
        self.conv6 = Conv2DTranspose(filters=1, kernel_size=(5, 5), activation='sigmoid')

    def call(self, z):
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        z = self.conv5(z)
        z = self.conv6(z)
        return z


class Critic(tf.keras.models.Model):

    def __init__(self):
        super(Critic, self).__init__()
        self.pooling = MaxPool2D(pool_size=(2, 2))
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.dense = Dense(16, activation='relu')
        self.dense = Dense(1, activation='linear')
        self.global_pooling = GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling(x)
        x = self.conv5(x)
        x = self.global_pooling(x)
        return x


def forward_generator(generator, batch_size):
    z = tf.random.normal(shape=(batch_size, 4, 4, 1))
    x = generator(z)
    return x

def train_generator(generator, critic, optimizer, batch_size):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(generator.trainable_weights)
        x = forward_generator(generator, batch_size)
        f = critic(x)
        loss = -tf.math.reduce_mean(f)
    grads = tape.gradient(loss, generator.trainable_weights)
    optimizer.apply_gradients(zip(grads, generator.trainable_weights))

def gradient_penalty(critic, x_prior, x_gen):
    batch_size = int(x_prior.shape[0])
    losses = []
    epsilon = tf.random.uniform(shape=[batch_size], minval=0., maxval=1., dtype=tf.float32)
    global_x_mixed = epsilon*x_prior  + (1. - epsilon)*x_gen
    for idx in range(batch_size):
        x_mixed = tf.expand_dims(global_x_mixed[idx, :, :, :], axis=0)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_mixed)
            f = critic(x_mixed)
        df_dx = tape.gradient(f, x_mixed)
        gnorm = tf.norm(df_dx)
        loss = (gnorm - 1.) ** 2.
        losses.append(loss)
    losses = tf.reduce_mean(losses)
    return losses

def train_discriminator(generator, critic, optimizer, prior_distribution, lbda):
    for x_prior in prior_distribution:
        batch_size = int(x_prior.shape[0])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(critic.trainable_weights)
            f_prior = critic(x_prior)
            x_gen = forward_generator(generator, batch_size)
            f_gen = critic(x_gen)
            wasserstein_loss = tf.reduce_mean(f_gen) - tf.reduce_mean(f_prior)
            gradient_loss = gradient_penalty(critic, x_prior, x_gen)
            loss = wasserstein_loss + gradient_loss
        grads = tape.gradient(loss, critic.trainable_weights)
        optimizer.apply_gradients(zip(grads, critic.trainable_weights))

def burn_in(generator, critic):
    x = forward_generator(generator, 4)
    f = critic(x)
    del f, x

def batched_prior(dataset, num_batchs):
    for batch in dataset.take(num_batchs):
        batch_image = batch['image']
        batch_image = tf.cast(batch_image, tf.float32)
        batch_image = batch_image / 255
        yield batch_image
    plt.show()

def load_cifar10(batch_size):
    dataset = tfds.load(name='mnist', split='train')
    dataset = dataset.shuffle(1024).repeat().batch(batch_size)
    return dataset

def evaluate_generator(generator, epoch):
    num_images = 10
    image = forward_generator(generator, num_images)
    for idx in range(num_images):
        fname = 'images/img_%d_%d.png'%(epoch+1, idx+1)
        plt.imsave(fname, np.squeeze(image.numpy()[idx,:,:,:]))

def train_wgan(generator, critic, dataset,
              batch_size, critic_steps_per_gen_steps,
              num_epochs, steps_per_epoch):
    for epoch in range(num_epochs):
        critic_opt = tf.keras.optimizers.Adam()
        gen_opt = tf.keras.optimizers.Adam()
        for _ in tqdm(range(steps_per_epoch), desc='Epoch %d/%d'%(epoch+1, num_epochs)):
            prior_distribution = batched_prior(dataset, critic_steps_per_gen_steps)
            train_discriminator(generator, critic, critic_opt, prior_distribution, lbda)
            train_generator(generator, critic, gen_opt, batch_size)
        evaluate_generator(generator, epoch)
        generator.save_weights('models/generator')
        critic.save_weights('models/critic')
        
        
if __name__ == '__main__':
    num_epochs = 1000
    steps_per_epoch = 100
    critic_steps_per_gen_steps = 6
    lbda = 10
    batch_size = 32
    generator = Generator()
    critic = Critic()
    burn_in(generator, critic)
    dataset = load_cifar10(batch_size)
    train_wgan(generator, critic, dataset,
               batch_size, critic_steps_per_gen_steps,
               num_epochs, steps_per_epoch)