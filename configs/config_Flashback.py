import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='NYC')
parser.add_argument('--base_epochs', type=int, default=20)
parser.add_argument('--incremental_epochs', type=int, default=10)
parser.add_argument('--mode', type=str, default='pretrain')

parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--pretrain_lr', type=float, default=3e-3)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--batch_size', type=int, default=400)

parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--n_samples', type=int, default=20)

args = parser.parse_args()
