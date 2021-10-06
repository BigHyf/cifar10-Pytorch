import argparse
from train import Trainer
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--model_name', default='my_model', type=str)
    parser.add_argument('--state_dir', default='./state', type=str)
    parser.add_argument('--log_dir', default='./log', type=str)
    parser.add_argument('--model', default='model', type=str)

    parser.add_argument('--num_epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--optimizer', type=str, default="sgd", help='choice of optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD optimizer momentum')

    parser.add_argument('--gpu', default='cuda:0', type=str)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
 