import argparse


def get_parser():
    # parser:
    parser = argparse.ArgumentParser(description='Evaluation Parser')

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
        "--test_dir",
        type=str,
        metavar='train_dir',
        required=True,
        help='Training directory for images'
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


    parser.add_argument(
        "--student_weights",
        type=str,
        metavar="teacher_weights",
        required=False,
        help="Directory of teacher weights"
    )


    return parser


if __name__ == '__main__':
    get_parser()
