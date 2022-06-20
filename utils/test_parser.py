import argparse


def get_parser():
    # parser:
    parser = argparse.ArgumentParser(description='Testing arguments')

    parser.add_argument('--device', type=int, default=0,
                        metavar='device', help='device used during training (default: 0)')

    parser.add_argument('--test_dir', type=str,
                        metavar='testing-directory', help='Directory of the testing csv', required=True)

    parser.add_argument('--weights_dir', type=str,
                        metavar='weights_dir', help='Directory of weights', required=True)

    parser.add_argument('--name', type=str,
                        metavar='name', help='Experiment name that logs into wandb')

    parser.add_argument('--project_name', type=str,
                        metavar='project_name', help='Project name, utilized for logging purposes in W&B.')

    parser.add_argument('--group', type=str,
                        metavar='group', help='Grouping argument for W&B init.')

    parser.add_argument('--workers', default=8,
                        metavar='workers', help='Number of workers for the dataloader')

    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        metavar='batch_size', help='input batch size for training (default: 32)')
    return parser


if __name__ == '__main__':
    get_parser()
