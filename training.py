import tensorflow as tf
import wandb

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from dataset import get_dataset
from network import network

AUTOTUNE = tf.data.experimental.AUTOTUNE
# tf.compat.v1.disable_eager_execution()  # causes compatibility issues with TF 2.x

train = True
saving = True

# Initialize wandb. Use disabled mode for debugging without logging.
# wandb.init(mode="disabled")
wandb.init(
    project="usiGrabber_AHLF",
    config={
        "channels": 64,
        "num_conv_layers": 13,
        "kernel_size": 2,
        "padding": "same",
        "dropout": 0.2,
        "learning_rate": 5.0e-6,
        "optimizer": "Adam",
        "loss": "BinaryCrossentropy",
        "batch_size": 64,
        "steps_per_epoch": 1000,
        "epochs": 10,
        "input_shape": [3600, 2]
    }
)

config = wandb.config

callbacks = [
    WandbMetricsLogger(log_freq=50),
    WandbModelCheckpoint("models")
]

ch = 64
net = network([ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch,ch],kernel_size=2,padding='same',dropout=.2) 

inp = tf.keras.layers.Input((3600,2))
sigm = net(inp)
model = tf.keras.Model(inputs=inp,outputs=sigm)
bce=tf.keras.losses.BinaryCrossentropy(from_logits=False)

learning_rate=5.0e-6
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0),loss=bce,metrics=['binary_accuracy','Recall','Precision'])

batch_size=64

steps=1000
maximum_steps=batch_size*steps
steps_per_epoch=steps

train_data = get_dataset(dataset=['./training'],maximum_steps=maximum_steps,batch_size=batch_size,mode='training').prefetch(buffer_size=AUTOTUNE)

if train:
    model.fit(train_data,steps_per_epoch=steps_per_epoch,epochs=10,callbacks=callbacks)

if saving:
    model.save_weights('model_weights_train2.hdf5')
