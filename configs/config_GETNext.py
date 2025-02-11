import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='NYC')
parser.add_argument('--base_epochs', type=int, default=50)
parser.add_argument('--incremental_epochs', type=int, default=15)
parser.add_argument('--mode', type=str, default='pretrain')

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--poi-embed-dim', type=int, default=128)
parser.add_argument('--user-embed-dim', type=int, default=128)
parser.add_argument('--gcn-dropout', type=float, default=0.3)
parser.add_argument('--gcn-nhid', type=list, default=[32, 64])
parser.add_argument('--transformer-nhid', type=int, default=1024)
parser.add_argument('--transformer-nlayers', type=int, default=2)
parser.add_argument('--transformer-nhead', type=int, default=2)
parser.add_argument('--transformer-dropout', type=float, default=0.3)
parser.add_argument('--time-embed-dim', type=int, default=32)
parser.add_argument('--cat-embed-dim', type=int, default=32)
parser.add_argument('--node-attn-nhid', type=int, default=128)
parser.add_argument('--time-loss-weight', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr_scheduler_factor', type=float, default=0.1)

parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--n_samples', type=int, default=20)

args = parser.parse_args()
