import tensorflow as tf
import wandb
import os
import numpy as np

from wandb.integration.keras import WandbMetricsLogger
from dataset import get_dataset
from network import network

AUTOTUNE = tf.data.experimental.AUTOTUNE

train = True

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
        "val_freq": 4000,
        "checkpoint_freq": 4000,
        "ion_current_normalize": "max",
    }
)

config = wandb.config

print(f"Using run name: {wandb.run.name}", flush=True)

class ValidationCallback(tf.keras.callbacks.Callback):
    """Run validation every val_freq training steps.

    Uses a manual forward pass instead of model.evaluate() to avoid
    resetting training metric accumulators
    """

    def __init__(self, val_data, val_freq):
        super().__init__()
        self.val_data = val_data
        self.val_freq = val_freq
        self.step_count = 0
        # Own metric instances — completely independent from model.fit() state
        self.val_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.val_metrics = {
            'loss': tf.keras.metrics.Mean(),
            'binary_accuracy': tf.keras.metrics.BinaryAccuracy(),
            'recall': tf.keras.metrics.Recall(),
            'precision': tf.keras.metrics.Precision(),
        }
    def on_train_batch_end(self, batch, logs=None):
        self.step_count += 1
        if self.step_count % self.val_freq == 0:
            # Reset our own val metrics
            for m in self.val_metrics.values():
                m.reset_states()

            phospho_correct = 0
            phospho_total = 0
            non_phospho_correct = 0
            non_phospho_total = 0

            i = 0
            for x_batch, y_batch in self.val_data:
                preds = self.model(x_batch, training=False)
                loss = self.val_loss_fn(y_batch, preds)
                self.val_metrics['loss'].update_state(loss)
                self.val_metrics['binary_accuracy'].update_state(y_batch, preds)
                self.val_metrics['recall'].update_state(y_batch, preds)
                self.val_metrics['precision'].update_state(y_batch, preds)
                y_np = y_batch.numpy().flatten()
                pred_np = (preds.numpy().flatten() >= 0.5).astype(float)
                phospho_idx = y_np == 1.0
                non_phospho_idx = y_np == 0.0
                phospho_correct += int(np.sum(pred_np[phospho_idx] == y_np[phospho_idx]))
                phospho_total += int(np.sum(phospho_idx))
                non_phospho_correct += int(np.sum(pred_np[non_phospho_idx] == y_np[non_phospho_idx]))
                non_phospho_total += int(np.sum(non_phospho_idx))
                i += 1

            print(f"Ran validation on {i} batches: {phospho_total} phospho, {non_phospho_total} non-phospho samples")

            val_results = {k: float(m.result()) for k, m in self.val_metrics.items()}
            val_results['acc_phospho'] = phospho_correct / phospho_total if phospho_total > 0 else float('nan')
            val_results['acc_non_phospho'] = non_phospho_correct / non_phospho_total if non_phospho_total > 0 else float('nan')

            val_log = {f'val_{k}': v for k, v in val_results.items()}
            val_log['val_step'] = self.step_count
            wandb.log(val_log, commit=False)
            print(f"\nStep {self.step_count} - Val: " +
                  ", ".join([f"{k}: {v:.4f}" for k, v in val_results.items()]))


class RawBatchLogger(tf.keras.callbacks.Callback):
    """Logs true per-batch metrics alongside the epoch-long smoothed average.

    WandbMetricsLogger logs Keras running averages — these reset at each epoch
    boundary and are dragged by early high-loss steps. This callback derives the
    raw per-batch loss from the running average and logs a short-window (log_freq)
    average as 'raw/loss' etc., giving an epoch-reset-free view of training.
    """

    def __init__(self, log_freq=50):
        super().__init__()
        self.log_freq = log_freq
        self._step_in_epoch = 0
        self._global_step = 0
        self._prev = {}       
        self._window = {}  

    def on_epoch_begin(self, epoch, logs=None):
        self._step_in_epoch = 0
        self._prev = {}
        self._window = {}

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            return
        self._step_in_epoch += 1
        self._global_step += 1
        n = self._step_in_epoch

        for key, running_avg in logs.items():
            prev = self._prev.get(key, running_avg)  # first step: treat prev = current
            per_batch = running_avg * n - prev * (n - 1)
            self._window.setdefault(key, []).append(per_batch)
            self._prev[key] = running_avg

        if self._step_in_epoch % self.log_freq == 0:
            raw_log = {f'raw/{k}': sum(v) / len(v) for k, v in self._window.items()}
            raw_log['raw/batch_step'] = self._global_step
            wandb.log(raw_log, commit=False)
            self._window = {}


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

data_path = ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training_shuffled_final/"]
validation_path = ["/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/validation_final/"]

wandb.run.config.data_path = data_path
wandb.run.config.validation_path = validation_path

train_data = get_dataset(
    dataset=data_path,
    batch_size=batch_size,
    mode='training',
).prefetch(buffer_size=AUTOTUNE)

val_data = get_dataset(
    dataset=validation_path,
    batch_size=batch_size,
    mode='test',
).prefetch(buffer_size=AUTOTUNE)

callbacks = [
    WandbMetricsLogger(log_freq=50),       # smoothed: epoch-long running avg → batch/*
    RawBatchLogger(log_freq=50),            # raw: 50-batch window avg → raw/*
    ValidationCallback(val_data, val_freq=config.val_freq),
    StepCheckpointCallback(checkpoint_freq=config.checkpoint_freq)
]

if train:
    model.fit(train_data, epochs=config.epochs, steps_per_epoch=37000, callbacks=callbacks)