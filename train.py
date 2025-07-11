import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2


def build_model(input_shape=(48, 48, 3)):
    """Build a CNN model using MobileNetV2 as a feature extractor."""
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(args):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    train_ds = train_gen.flow_from_directory(
        directory=args.train_dir,
        target_size=(48, 48),
        batch_size=args.batch_size,
        class_mode='categorical'
    )
    val_ds = train_gen.flow_from_directory(
        directory=args.val_dir,
        target_size=(48, 48),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(args.output)

    # Optionally save training history
    if args.history:
        import json
        with open(args.history, 'w') as f:
            json.dump(history.history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion detection model')
    parser.add_argument('--train-dir', required=True,
                        help='Path to training images directory')
    parser.add_argument('--val-dir', required=True,
                        help='Path to validation images directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output', default='model.h5',
                        help='Path where the trained model will be saved')
    parser.add_argument('--history', default='',
                        help='Optional path to save training history json')
    main(parser.parse_args())
