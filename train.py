import os
import sys
from datetime import datetime
from os.path import join
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.optimizers import Adam
import mlflow
from utils.callbacks import get_callbacks
from utils.mlflow_handler import MLFlowHandler
from utils.utils import get_gpu_grower
from utils.plots import get_plots

get_gpu_grower()


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")

    id_ = model_name + "_" + str(datetime.now().date()) + "_" + str(datetime.now().time())
    # file with ':' in their name  not allowed in Windows.
    id_ = id_.replace(':', '.')
    weight_path = os.path.join(os.getcwd(), 'weights', id_) + ".h5"
    mlflow_handler = MLFlowHandler(model_name=model_name, run_name=id_, mlflow_source=args.mlflow_source)
    mlflow_handler.start_run(args)

    # Loading Data
    train_loader, valid_loader, test_loader = get_loader(train_path=args.train_path,
                                                         val_path=args.val_path,
                                                         test_path=args.test_path,
                                                         batch_size=args.batch_size,
                                                         target_size=tuple(args.target_size)
                                                         )
    print("Loading Data is Done!")

    # Loading Model
    model = load_model(model_name=model_name,
                       image_size=args.target_size,
                       n_classes=args.n_classes,
                       fine_tune=args.fine_tune
                       )
    print("Loading Model is Done!")
    checkpoint, reduce_lr, early_stopping = get_callbacks(weight_path,
                                                          early_stopping_p=5,
                                                          mlflow=mlflow)

    # -------------------------------------------------------------------

    # Training
    opt = Adam(learning_rate=args.lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',  # arg
                  metrics=['acc']
                  )
    print("Training Model...")
    model.fit(train_loader,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=valid_loader,
              validation_batch_size=args.batch_size,
              callbacks=[checkpoint, reduce_lr, mlflow_handler.mlflow_logger]
              )
    print("Training Model is Done!")

    get_plots(model, test_loader, args, mlflow_handler)
    mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()
