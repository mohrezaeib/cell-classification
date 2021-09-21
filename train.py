from models import load_model
from params import get_args
from data.data_loader import get_loader


from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import mlflow


def train():
    model_name = sys.argv[2]
    # model_name = 'resnet50'
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")

    # Loading Data
    train_loader, valid_loader, test_loader = get_loader(args.dataset_path,  # dataset dir path
                                                         args.valid_size,
                                                         args.batch_size,
                                                         tuple(args.target_size)
                                                         # image size that generated by data loader
                                                         )
    n_classes = args.n_classes
    print("Loading Data is Done!")

    # Loading Model
    model = load_model(model_name=model_name,
                       image_size=args.target_size,
                       n_classes=args.n_classes,
                       fine_tune=args.fine_tune
                       )
    print("Loading Model is Done!")

    # -------------------------------------------------------------------
    # Class weights
    y_train = train_loader.classes
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y_train),
                                                      y_train)
    class_weights = dict(enumerate(class_weights))
    # print(f'Class weights: {class_weights}')

    # Call Backs
    checkpoint_path = ''
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=False,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.9,  # new_lr = lr * factor
                                  patience=10,  # number of epochs with no improvment
                                  min_lr=0.0001,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )

    # -------------------------------------------------------------------
    mlflow.set_experiment(str(model_name))
    experiment = mlflow.get_experiment_by_name(str(model_name))
    ex_id = experiment.experiment_id

    with mlflow.start_run(run_name=str(model_name), experiment_id=str(ex_id)) as run:

        # Training
        opt = Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy', # arg
                      metrics=['acc']
                      )
        print("Training Model...")
        history = model.fit(train_loader,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            validation_data=valid_loader,
                            validation_batch_size=args.batch_size,
                            # class_weight=class_weights,
                            callbacks=[checkpoint, reduce_lr]
                            )
        print("Training Model is Done!")

        mlflow.log_param('batch size', args.batch_size)
        mlflow.log_param('validation batch size', args.val_batch_size)
        mlflow.log_param('loss', 'Binary Cross-entropy')
        mlflow.log_param('epochs', args.epochs)

        mlflow.log_metric('train acc', history.history['acc'][0])
        mlflow.log_metric('val acc', history.history['val_acc'][0])
        mlflow.log_metric('train loss', history.history['loss'][0])
        mlflow.log_metric('val_loss', history.history['val_loss'][0])

        mlflow.log_artifact()

    # -------------------------------------------------------------------
    # Evaluation
    print('Evaluation')
    predictions = model.predict(test_loader, steps=test_loader.n // args.batch_size + 1)
    y_pred = np.argmax(predictions, axis=-1)
    y_true = test_loader.classes
    # print(f'y_pred: {len(y_preds)}')
    # print(f'y_true: {len(y_true)}')

    # Metrics: Train: Loss plot
    plt.plot(history.history['loss'], 'b-', label="Train")
    plt.plot(history.history['val_loss'], 'r-', label="Valid")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # Metrics: Train: ACC plot
    plt.plot(history.history['acc'], 'b-', label="Train")
    plt.plot(history.history['val_acc'], 'r-', label="Valid")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # Metrics: Test: Loss, Acc
    test_score = model.evaluate(test_loader, steps=test_loader.n // args.batch_size + 1)  # test data
    print(f'Test: loss= {test_score[0]}, Accuracy: {test_score[1]}')

    # Metrics : precision, recall, f1-score
    print(classification_report(y_true, y_pred))

    # Metrics: Confusion Matrix
    con_mat = confusion_matrix(y_true, y_pred)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_classes)], columns=[i for i in range(n_classes)])
    figure = plt.figure(figsize=(n_classes, n_classes))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return run.info.experiment_id, run.info.run_id


if __name__ == '__main__':
    train()