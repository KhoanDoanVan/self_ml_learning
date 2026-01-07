import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# Path to json file that stores MFCCs and Genre labels for each processed segment
DATA_PATH = "path/to/dataset"

def load_data(
        data_path
):
    """
    Loads training dataset from json file
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Outputs
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y



def plot_history(
        history
):
    """
    Plot accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # Create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="Train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="Test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")

    # Create error subplot
    axs[1].plot(history.history["loss"], label="Train error")
    axs[1].plot(history.history["val_loss"], label="Test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")

    plt.show()



if __name__ == "__main__":

    # Load data
    X, y = load_data(DATA_PATH)

    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3
    )

    # Build network topology
    model = keras.Sequential([

        # input layer
        keras.layers.Flatten(
            input_shape=(
                X.shape[1],
                X.shape[2]
            )
        ),

        # 1st Dense Layer
        keras.layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),

        keras.layers.Dropout(0.3),

        # 2nd Dense Layer
        keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),

        keras.layers.Dropout(0.3),

        # 3rd Dense Layer
        keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.001)
        ),

        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])


    # Optimizer GD
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001
    )

    # Compile
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )


    model.summary()

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_test,
            y_test
        ),
        batch_size=32,
        epochs=100
    )

    # Plot Accuracy and Error as a function of the epochs
    plot_history(history)

