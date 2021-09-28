from argparse import ArgumentParser


def main_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name. Default = resnet50', required=True)
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Define the size of every training batch. Default = 32')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Choose verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default = 1')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Choose the training optimizer. Default = adam')
    parser.add_argument('--min_lr', type=float, default=0.001,
                        help='min_lr learning rate. Default is 0.001')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience in learning rate schedule. Default is 10')
    parser.add_argument('--epochs', type=int, default=10, help='Define number of training epochs.', required=False)
    parser.add_argument('--train-path', type=str, default='data/data/train',
                        help='Path to folder containing train dataset directory.',
                        required=False)
    parser.add_argument('--val-path', type=str, default='data/data/val',
                        help='Path to folder containing val dataset directory.',
                        required=False)
    parser.add_argument('--test-path', type=str, default='data/data/test',
                        help='Path to folder containing test dataset directory.',
                        required=False)
    parser.add_argument('--target-size', type=list, nargs=2, default=[224, 224], help='Image size for model.',
                        required=False)
    parser.add_argument('--n-classes', type=int, default=4, help='Define number of classes', required=False)
    parser.add_argument('--fine_tune', type=bool, default=False, help='To Fine-tune Set True', required=False)
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate.', required=False)
    parser.add_argument('--model_path', type=str, default="weights", help='model path', required=False)
    parser.add_argument('--aug_prob', type=float, default=0.5, help='The probability of applying augmentations')
    parser.add_argument('--mlflow-source', type=str, default='./mlruns', help='The mlflow direcotry')
    return parser
