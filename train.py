import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten,
                                     Dense, BatchNormalization)
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


def build_model(input_shape=(48, 48, 3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main(args):
    train_gen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   vertical_flip=True)
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
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
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
