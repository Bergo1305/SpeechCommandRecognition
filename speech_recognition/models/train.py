from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from speech_recognition import architectures
from speech_recognition.optimizers import Optimizers
from speech_recognition.generators.common import CSVDataGenerator
from speech_recognition.utils.visualizer import ModelVisualizer
from config import trainingConfiguration, logger, CURRENT_DIR

logger.info(
    "*******************TRAINING CONFIGURATION********************"
)
logger.info(
    {
        f"BATCH-SIZE: {trainingConfiguration.batch_size}",
        f"NUM_CLASSES: {trainingConfiguration.num_classes}",
        f"SAMPLING_RATE: {trainingConfiguration.sampling_rate}",
        f"EPOCHS: {trainingConfiguration.num_epochs}",
        f"OPTIMIZER: {trainingConfiguration.optimizer}",
        f"DECAY_RATE: {str(trainingConfiguration.decay_rate)}"
    }
)


if __name__ == "__main__":

    callbacks = []

    train_generator = CSVDataGenerator(
        f"{CURRENT_DIR}/dataset/train/train.csv",
        batch_size=trainingConfiguration.batch_size,
        sampling_rate=trainingConfiguration.sampling_rate,
        n_classes=trainingConfiguration.num_classes
    )

    validation_generator = CSVDataGenerator(
        f"{CURRENT_DIR}/dataset/train/validation.csv",
        batch_size=trainingConfiguration.batch_size,
        sampling_rate=trainingConfiguration.sampling_rate,
        n_classes=trainingConfiguration.num_classes
    )

    if trainingConfiguration.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=10,
            min_delta=0.0001
        )
        callbacks.append(early_stopping)

    if trainingConfiguration.model_checkpoint:
        model_checkpoint = ModelCheckpoint(
            "best_model.h5",
            monitor="val_accuracy",
            verbose=1,
            save_best_only=False,
            mode="max"
        )
        callbacks.append(model_checkpoint)

    optimizer = getattr(Optimizers, trainingConfiguration.optimizer.lower())(
        learning_rate=trainingConfiguration.learning_rate
    )

    model = getattr(architectures, trainingConfiguration.base_model)(
        trainingConfiguration.num_classes
    )

    model.compile(
        optimizer=optimizer,
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    history = model.fit_generator(
        generator=train_generator,
        epochs=trainingConfiguration.num_epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    model_visualizer = ModelVisualizer(history)
    model_visualizer.plot(
        accuracy_img_path="train/val-accuracy.png",
        loss_img_path="train/val-loss.png"
    )



