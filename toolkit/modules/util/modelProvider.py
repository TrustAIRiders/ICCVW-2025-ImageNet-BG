import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def remove_softmax(model):
    assert model.layers[-1].activation == tf.keras.activations.softmax

    config = model.layers[-1].get_config()
    weights = [x.numpy() for x in model.layers[-1].weights]

    config['activation'] = tf.keras.activations.linear
    config['name'] = 'logits'

    new_layer = tf.keras.layers.Dense(**config)(model.layers[-2].output)
    model_wo_softmax = tf.keras.Model(inputs=[model.input], outputs=[new_layer])
    model_wo_softmax.layers[-1].set_weights(weights)

    assert model_wo_softmax.layers[-1].activation == tf.keras.activations.linear

    return model_wo_softmax

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

model_wo_softmax = remove_softmax(model)

def eval(image):
    return model(image[None, :, :, :])

def eval_wo_softmax(image):
    return model_wo_softmax(image[None, :, :, :])