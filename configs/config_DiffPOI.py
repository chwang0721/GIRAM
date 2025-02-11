import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='NYC')
parser.add_argument('--base_epochs', type=int, default=30)
parser.add_argument('--incremental_epochs', type=int, default=10)
parser.add_argument('--b', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1024,
                    help="the batch size for training procedure")
parser.add_argument('--beta', type=float, default=0.2,
                    help="fisher loss weight")
parser.add_argument('--hidden', type=int, default=64,
                    help="node embedding size")
parser.add_argument('--interval', type=int, default=256,
                    help="types of temporal and locational intervals")
parser.add_argument('--num_layer', type=int, default=2,
                    help="layer num of GNN")
parser.add_argument('--diffsize', type=int, default=1,
                    help="diffusion size T")
parser.add_argument('--stepsize', type=float, default=0.1,
                    help="diffusion step size dt")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate")
parser.add_argument('--decay', type=float, default=1e-3,
                    help="weight decay for l2 normalizaton")
parser.add_argument('--dropout', action='store_true', default=False,
                    help="using the dropout or not")
parser.add_argument('--keepprob', type=float, default=0.6,
                    help="dropout probalitity")
parser.add_argument('--patience', type=int, default=10,
                    help="early stop patience")
parser.add_argument('--device', type=str, default='cuda:0',
                    help='training device')
parser.add_argument('--mode', type=str, default='pretrain',
                    help='training mode')
parser.add_argument('--n_samples', type=int, default=20)
args = parser.parse_args()
