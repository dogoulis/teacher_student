import argparse


def get_parser():
    # parser:
    parser = argparse.ArgumentParser(description='Student arguments')

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model used for the experiment.'
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:1',
        metavar="device",
        help="device used during training (default: 0)",
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        metavar='train_dir',
        required=True,
        help='Training directory for images'
    )

    parser.add_argument(
        "--valid_dir",
        type=str,
        metavar='valid_dir',
        required=True,
        help='Validation directory for images'
    )

    # WANDB
    parser.add_argument(
        "--name", type=str, metavar="name", help="Experiment name that logs into wandb"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        metavar="project_name",
        help="Project name, utilized for logging purposes in W&B.",
    )

    parser.add_argument(
        "--group", type=str, metavar="group", help="Grouping argument for W&B init."
    )

    # dataset

    parser.add_argument(
        "--workers",
        default=8,
        metavar="workers",
        help="Number of workers for the dataloader",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        metavar="batch_size",
        help="input batch size for training (default: 32)",
    )

    # training

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=20,
        required=False,
        metavar="epochs",
        help="Max number of epochs to train for",
    )

    # OPTIMZER
    parser.add_argument(
        "-opt",
        "--optimizer",
        type=str,
        default="adam",
        required=False,
        metavar="optimizer",
        help="optimizer to use during training (default: adam).",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        required=False,
        metavar="learning_rate",
        help="Learning rate of the optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=1e-5,
        required=False,
        metavar="weight_decay",
        help="Weight decay of the optimizer (default: 1e-5).",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=0.,
        required=False,
        metavar="momentum",
        help="Momentum of the optimizer (default: 0.).",
    )
    # SCHEDULE
    parser.add_argument(
        "-sch",
        "--scheduler",
        type=str,
        default="steplr",
        required=False,
        metavar="scheduler",
        help="Scheduler to use during training (default: steprl).",
    )
    parser.add_argument(
        "-step",
        "--scheduler_step_size",
        type=int,
        default=5,
        required=False,
        metavar="scheduler_step_size",
        help="scheduler step size (default: 5)",
    )
    parser.add_argument(
        "-gamma",
        "--scheduler_gamma",
        type=float,
        default=0.1,
        required=False,
        metavar="scheduler_gamma",
        help="scheduler gamma (default: 0.1)",
    )
    # add int argument for tmult with default value of 2
    parser.add_argument(
        "-tmult",
        "--tmult",
        type=int,
        default=2,
        required=False,
        metavar="tmult",
        help="tmult (default: 2)",
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        required=True,
        metavar='save_model_path',
        help='saving directory'
    )

    parser.add_argument(
        "--volume_loss",
        type=str,
        metavar="volume_loss",
        required=False,
        help='Flag for applying loss to 3d volumes'
    )

    parser.add_argument(
        "--teacher_weights",
        type=str,
        metavar="teacher_weights",
        required=True,
        help="Directory of teacher weights"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        metavar='alpha',
        default=0.5,
        help="Alpha value for weighting loss"
    )
    return parser


if __name__ == '__main__':
    get_parser()
