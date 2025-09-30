import math
from utils import display
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
activations = keras.activations
metrics = keras.metrics
register_keras_serializable = keras.utils.register_keras_serializable
callbacks = keras.callbacks
optimizers = keras.optimizers

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

@register_keras_serializable(package="diffusion")
def sinusoidal_embedding(x, noise_embedding_size: int):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            noise_embedding_size // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings

def get_unet(image_size: int, noise_embedding_size: int, num_channels: int = 1):
    noisy_images = layers.Input(shape=(image_size, image_size, num_channels))

    # Première projection des images
    x = layers.Conv2D(32, kernel_size=1)(noisy_images)

    # Embedding sinusoidal
    noise_variances = layers.Input(shape=(1, 1, 1))
    noise_embedding = layers.Lambda(
        sinusoidal_embedding,
        arguments={"noise_embedding_size": noise_embedding_size}
    )(noise_variances)
    noise_embedding = layers.UpSampling2D(
        size=image_size,
        interpolation="nearest"
    )(noise_embedding)

    # Projection de l’embedding pour matcher la largeur des features
    noise_embedding = layers.Conv2D(32, kernel_size=1)(noise_embedding)

    # Concaténation puis projection pour revenir à 32 canaux
    x = layers.Concatenate()([x, noise_embedding])
    x = layers.Conv2D(32, kernel_size=1)(x)

    skips = []
    x = DownBlock(32, block_depth=2)([x, skips])
    x = DownBlock(64, block_depth=2)([x, skips])
    x = DownBlock(96, block_depth=2)([x, skips])

    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)

    x = UpBlock(96, block_depth=2)([x, skips])
    x = UpBlock(64, block_depth=2)([x, skips])
    x = UpBlock(32, block_depth=2)([x, skips])

    x = layers.Conv2D(num_channels, kernel_size=1, kernel_initializer="zeros")(x)

    unet = models.Model([noisy_images, noise_variances], x, name="unet")
    return unet



def offset_cosine_diffusion_schedule(diffusion_times):
    """
    Implements the cosine diffusion schedule with small offset to avoid
    extremely small noise rates.
    """
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

# Callbacks
class ImageGenerator(callbacks.Callback):
    """Generates and saves sample images after every epoch."""
    def __init__(self, num_img, plot_diffusion_steps: int):
        self.num_img = num_img
        self.plot_diffusion_steps = plot_diffusion_steps

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generate(
            num_images=self.num_img,
            diffusion_steps=self.plot_diffusion_steps,
        ).numpy()
        display(
            generated_images,
            save_to=f"./output/generated_img_{epoch:03d}.png",
        )

class DiffusionModel(models.Model):
    def __init__(self,
                 image_size: int,
                 num_channels: int,
                 noise_embedding_size: int,
                 batch_size: int,
                 ema: float = 0.995,
                 load_weights_path: str = None,
                 plot_diffusion_steps: int = 20,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,):
        """Implements a diffusion model with a U-Net architecture.
        

        Parameters
        ----------
        image_size: int
            The size of the input images (image_size x image_size).
        num_channels: int
            The number of channels in the input images (e.g., 3 for RGB, 1 for grayscale).
        noise_embedding_size: int
            The size of the noise embedding vector.
        batch_size: int
            The batch size for training.
        ema: float
            The exponential moving average factor for the model weights.
        load_weights_path: str
            Path to load pre-trained weights. If None, weights are initialized randomly.
        plot_diffusion_steps: int
            The number of diffusion steps to use when generating sample images.
        learning_rate: float
            The learning rate for the optimizer.
        weight_decay: float
            The weight decay for the optimizer.
        """
        super().__init__()
        # parameters
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.ema = ema
        self.plot_diffusion_steps = plot_diffusion_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # model components
        self.normalizer = layers.Normalization(axis=-1) # normalize images to mean 0 and variance 1 (axis=-1 for channels last)
        self.network = get_unet(image_size, noise_embedding_size, num_channels=num_channels)
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

        # callbacks
        self.callbacks_list = [
            ImageGenerator(num_img=5, plot_diffusion_steps=self.plot_diffusion_steps),
            callbacks.ModelCheckpoint(
                        filepath="./checkpoint/ckpt.weights.h5",
                        save_weights_only=True,
                        save_freq="epoch",
                        verbose=0,
                    ),
            callbacks.TensorBoard(log_dir="./logs"),
            ]
        
        # load weights if a path is provided
        self.load_weights_path = load_weights_path

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    def build(self, input_shape):
        if self.load_weights_path is not None:
            self.built = True
            self.load_weights(self.load_weights_path)
        super().build(input_shape)

    def fit(self, dataset, epochs):
        return super().fit(
            dataset, 
            epochs=epochs, 
            callbacks=self.callbacks_list
            )
    
    def train(self, dataset: tf.data.Dataset, epochs: int = 1):
        self.normalizer.adapt(dataset)
        self.compile(
            optimizer=optimizers.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
            ),
            loss=keras.losses.MeanSquaredError(),
        )
        self.build((self.batch_size, self.image_size, self.image_size, self.num_channels))
        self.fit(dataset, epochs=epochs)

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates**2], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, self.image_size, self.image_size, self.num_channels),
            )
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
            )
        else:
            # ensure we work with tf tensors (no mixing numpy arrays + tf tensors)
            initial_noise = tf.convert_to_tensor(initial_noise, dtype=tf.float32)
            # keep num_images consistent with the provided initial_noise
            num_images = int(initial_noise.shape[0])
            generated_images = self.reverse_diffusion(
                initial_noise, diffusion_steps
                )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))

        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises, pred_noises)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}