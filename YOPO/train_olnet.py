import os
import torch
import random
import argparse
import numpy as np
from policy.olnet_trainer import OLNetTrainer


def configure_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=int, default=0, help="use pre-trained model?")
    parser.add_argument("--trial", type=int, default=1, help="trial of pre-trained model")
    parser.add_argument("--epoch", type=int, default=50, help="epoch of pre-trained model")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    configure_random_seed(0)    # set random seed

    # save the configuration and other files
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/saved"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = log_dir + "/OLNET_{}/epoch_{}.pth".format(args.trial, args.epoch) if args.pretrained else ""

    trainer = OLNetTrainer(
        learning_rate=0.001,
        batch_size=32,
        tensorboard_path=log_dir,
        checkpoint_path=checkpoint_path,
        save_on_exit=True,
    )

    trainer.train(epoch=50)

    print("Run OLNet Finish!")
