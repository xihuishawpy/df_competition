import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Add,
    Concatenate,
    Layer,
)
from tensorflow.keras.losses import (
    sparse_categorical_crossentropy,
    binary_crossentropy,
    mse,
    mae,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from deeptabular.transformer import EncoderLayer


class AttentionPooling(Layer):
    def __init__(self, d_model, name=None):
        super(AttentionPooling, self).__init__(name=name)
        self.d_model = d_model

        self.wq = Dense(d_model, activation="relu")
        self.d_score = Dense(1)

    def call(self, q, mask=None):
        batch_size = tf.shape(q)[0]

        proj_q = self.wq(q)  # (batch_size, seq_len, d_model)
        scores = self.d_score(proj_q)  # (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)

        matmul_attention = tf.matmul(
            q, attention_weights, transpose_b=False, transpose_a=True
        )  # (batch_size, d_model, 1)

        concat_attention = tf.reshape(
            matmul_attention, (batch_size, self.d_model)
        )  # (batch_size, d_model)

        return concat_attention


def transformer_tabular(
    n_categories,
    n_targets,
    embeds_size=10,
    num_layers=4,
    num_dense_layers=4,
    d_model=64,
    num_heads=4,
    task="classification",
    lr=0.0001,
    dropout=0.01,
    seq_len=None,
    att_heads=2,
):
    dff = 2 * d_model
    input_cols = Input(shape=(seq_len,))
    input_values = Input(shape=(seq_len, 1))

    x1 = Embedding(n_categories, embeds_size, name="embed")(input_cols)
    x2 = Dense(embeds_size, activation="linear", name="d1")(input_values)

    x = Add()([x1, x2])

    x = Dense(d_model, activation="relu", name="d2")(x)

    l_encoded = []

    for i in range(num_layers):
        x = EncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            rate=dropout,
            name="encoder_%s" % i,
        )(x)
        l_encoded.append(x)

    x_encoded = l_encoded[-1]

    if task in ["classification", "regression", "multilabel"]:

        x = Concatenate()(
            [
                AttentionPooling(d_model=d_model, name="att_%s" % i)(a)
                for i, a in enumerate(l_encoded * att_heads)
            ]
        )

        x = Dense(max(d_model, n_targets), name="d3", activation="relu")(x)

        for i in range(num_dense_layers):
            x_to_add = Dense(
                max(d_model, n_targets), activation="relu", name="d4_%s" % i
            )(x)
            x = Add()([x, x_to_add])

    if task == "classification":
        if n_targets > 1:

            out = Dense(n_targets, activation="softmax", name="d5")(x)

        else:

            out = Dense(n_targets, activation="sigmoid", name="d6")(x)

    elif task == "regression":
        out = Dense(n_targets, activation="linear", name="d7")(x)

    elif task == "multilabel":
        out = Dense(n_targets, activation="sigmoid", name="d9")(x)

    elif task == "pretrain":
        out = Dense(n_targets, activation="linear", name="d8")(x_encoded)

    else:
        raise NotImplementedError

    model = Model(inputs=[input_cols, input_values], outputs=out)

    opt = Adam(lr)

    if task == "classification":

        if n_targets > 1:

            model.compile(
                optimizer=opt, loss=sparse_categorical_crossentropy, metrics=["acc"]
            )

        else:

            model.compile(optimizer=opt, loss=binary_crossentropy, metrics=["acc"])

    elif task == "regression":

        model.compile(optimizer=opt, loss=mse)

    elif task == "multilabel":

        model.compile(optimizer=opt, loss=binary_crossentropy)

    elif task == "pretrain":
        model.compile(optimizer=opt, loss=mae)

    model.summary()

    return model
