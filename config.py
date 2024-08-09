import argparse
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Fairness TAG')
parser.add_argument('--input-dir', type=str, default='./data/')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--in-dim', type=int, default=200)
parser.add_argument('--out-dim', type=int, default=512)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--hidden-size', type=int, default=200)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--n-layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--conv-norm', type=bool, default=True)
parser.add_argument('--conv-name', type=str, default='gcn')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--mode', type=str, default='ft_lm', choices=['gnn', 'ft_lm', 'po'])
parser.add_argument('--output-dir', type=str, default='./checkpoints')

# lm config
parser.add_argument('--plm-path', type=str, default='/root/autodl-tmp/models/AI-ModelScope/bert-base-uncased')
parser.add_argument('--plm-finetune', action='store_true')
parser.add_argument('--pooling', type=str, default='mean', choices=['max', 'mean'])
parser.add_argument('--lm-batch-size', type=int, default=8)
parser.add_argument('--lm-epochs', type=int, default=2)
parser.add_argument('--sm-batch-size', type=int, default=2)
parser.add_argument('--infer-batch-size', type=int, default=16)
parser.add_argument('--ft-lr', type=float, default=1e-5)
# peft config
parser.add_argument('--peft-r', type=int, default=8)
parser.add_argument('--peft-lora-alpha', type=int, default=8)
parser.add_argument('--peft-lora-dropout', type=float, default=0.2)
parser.add_argument('--use-peft', action='store_true')
parser.add_argument('--use-full', action='store_true')
parser.add_argument('--eval-steps', type=int, default=500)
parser.add_argument('--logging-steps', type=int, default=50)

# selective droping 
parser.add_argument('--filter', action='store_true')
parser.add_argument('--add-kl', action='store_true')
parser.add_argument('--oracle-sm-batch-size', type=int, default=2)
parser.add_argument('--oracle-batch-size', type=int, default=8)
parser.add_argument('--oracle-model-path', type=str, default='/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased/save_model')

# po
# parser.add_argument('--po', action='store_true')
parser.add_argument('--ref-model-path', type=str, default='/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased_filter/save_model')
parser.add_argument('--po-sm-batch-size', type=int, default=2, )
parser.add_argument('--po-batch-size', type=int, default=8)
parser.add_argument('--po-lr', type=float, default=1e-5)
parser.add_argument('--po-epoch', type=int, default=2)
parser.add_argument('--po-beta', type=float, default=0.1)

args = parser.parse_args()
args.plm_name = args.plm_path[args.plm_path.rfind('/') + 1:]
 