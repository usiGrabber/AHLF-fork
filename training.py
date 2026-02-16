import tensorflow as tf
import wandb
import os

from wandb.integration.keras import WandbMetricsLogger
from dataset import get_dataset
from network import network

AUTOTUNE = tf.data.experimental.AUTOTUNE
# tf.compat.v1.disable_eager_execution()  # causes compatibility issues with TF 2.x

train = True
saving = True

# Initialize wandb. Use disabled mode for debugging without logging.
# wandb.init(mode="disabled")
wandb.init(
    entity="mp2025-usigrabber",
    project="ahlf-training",
    config={
        "channels": 64,
        "num_conv_layers": 13,
        "kernel_size": 2,
        "padding": "same",
        "dropout": 0.2,
        "learning_rate": 5.0e-4,
        "optimizer": "Adam",
        "loss": "BinaryCrossentropy",
        "batch_size": 128,
        "epochs": 10,
        "input_shape": [3600, 2],
        "val_freq": 2000,
        "checkpoint_freq": 4000,
        "val_ratio": 0.1,
        "ion_current_normalize": "matthis",
        "is_balanced": False
    }
)

config = wandb.config

print(f"Using run name: {wandb.run.name}", flush=True)

class ValidationCallback(tf.keras.callbacks.Callback):
    """Run validation every val_freq training steps."""

    def __init__(self, val_data, val_freq):
        super().__init__()
        self.val_data = val_data
        self.val_freq = val_freq
        self.step_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.val_freq == 0:
            val_results = self.model.evaluate(
                self.val_data,
                verbose=0,
                return_dict=True
            )
            # Log with val_ prefix to wandb
            wandb.log({f'val_{k}': v for k, v in val_results.items()}, step=self.step_count)
            print(f"\nStep {self.step_count} - Val: " +
                  ", ".join([f"{k}: {v:.4f}" for k, v in val_results.items()]))


class StepCheckpointCallback(tf.keras.callbacks.Callback):
    """Save checkpoint every checkpoint_freq training steps."""

    def __init__(self, checkpoint_freq, checkpoint_dir='checkpoints'):
        super().__init__()
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.step_count = 0
        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.checkpoint_freq == 0:
            path = os.path.join(self.checkpoint_dir, f'{wandb.run.name}-step_{self.step_count}.weights.hdf5')
            self.model.save_weights(path)
            # Log as wandb artifact
            artifact = wandb.Artifact(f'checkpoint-step-{self.step_count}', type='model')
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            print(f"\nCheckpoint saved: {path}")

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=.2) 

inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)

learning_rate=config.learning_rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0),loss=bce,metrics=['binary_accuracy','Recall','Precision'])

batch_size=config.batch_size
# Total samples in training directory (50/50 phospho/non-phospho)
# num_samples= 6596016 * 2
# # Adjust for train/val split - only (1 - val_ratio) of samples go to training
# steps_per_epoch = int(num_samples * (1 - config.val_ratio)) // batch_size

# dataset_path_list = "/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training_shuffled/bucket0.txt"

# with open(dataset_path_list, "r") as f:
#     data_path = f.readlines()

# data_path = ['/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training_shuffled/1/']
data_path = ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/pass_threshold_shuffled/"]
validation_path = ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/validation/"]

wandb.run.config.data_path = data_path
wandb.run.config.validation_path = validation_path

train_data = get_dataset(
    dataset=data_path,
    batch_size=batch_size,
    mode='training',
    is_balanced=config.is_balanced
).prefetch(buffer_size=AUTOTUNE)

val_data = get_dataset(
    dataset=validation_path,
    batch_size=batch_size,
    mode='test',
    is_balanced=True
).prefetch(buffer_size=AUTOTUNE)

callbacks = [
    WandbMetricsLogger(log_freq=50),
    ValidationCallback(val_data, val_freq=config.val_freq),
    StepCheckpointCallback(checkpoint_freq=config.checkpoint_freq)
]

if train:
    model.fit(train_data, epochs=40, callbacks=callbacks)

if saving:
    model.save_weights('model_weights_train2.hdf5')
