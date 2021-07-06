"""
    try to use GAN
    Generator: input noise (random vector with no bias) out put feature
    Discriminator: input feature from Generator output fake or real (binary classifier)
"""
import matplotlib.pyplot as plt
import tensorflow as tf

# load dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# process data
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
BATCH_SIZE = 256
BUFFER_SIZE = 60000
datasets = tf.data.Dataset.from_tensor_slices(train_images)
datasets = datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(100, ), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(512, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(28*28*1, use_bias=False, activation='sigmoid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Reshape((28, 28, 1))
    ])

    return model


def discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(256, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(1),
    ])

    return model


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# we hope discriminator output fake
def discriminator_loss(real, fake):
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)

    return real_loss + fake_loss


# we hope generator output true
def generator_loss(fake):
    return loss(tf.ones_like(fake), fake)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# init
noise_dim = 100
generator = generator_model()
discriminator = discriminator_model()

# visualize
num_example_to_generate = 16
seed = tf.random.normal([num_example_to_generate, noise_dim])


def train_step(images):
    noise = tf.random.normal([256, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        real_out = discriminator(images, training=True)

        gen_image = generator(noise, training=True)
        fake_out = discriminator(gen_image, training=True)

        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))


def generator_plot_image(gen_model, test_noise):
    pre_images = gen_model(test_noise, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(pre_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        print('epoch {}'.format(epoch+1))
        for image_batch in dataset:
            train_step(image_batch)
            print('#', end='')
        generator_plot_image(generator, seed)


# all
train(datasets, epochs=100)
