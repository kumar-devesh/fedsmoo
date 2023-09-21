from utils_libs import *
from utils_dataset import *
from utils_models import *
from methods import *

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"

#### ================= Open Float32 in A100 ================= ####
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#### ================= Open ignore warining ================= ####
import warnings
warnings.filterwarnings('ignore')
#### ======================================================== ####

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'tinyimagenet'], type=str, default='CIFAR10')
parser.add_argument('--model', choices=['ResNet18'], type=str, default='ResNet18')
parser.add_argument('--non-iid', action='store_true', default=False)
parser.add_argument('--rule', choices=['Drichlet', 'Path'], type=str, default='Path')
parser.add_argument('--rule-arg', default=3, type=float)
parser.add_argument('--act_prob', default=0.1, type=float)
parser.add_argument('--method', choices=['FedAvg', 'FedDyn', 'FedSMOO', 'MoFedSAM'], type=str, default='FedGS')
parser.add_argument('--n_client', default=100, type=int)
parser.add_argument('--epochs', default=800, type=int)
parser.add_argument('--local-epochs', default=5, type=int)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--alpha-coef', default=0.1, type=float)
parser.add_argument('--gamma-coef', default=0.0, type=float)
parser.add_argument('--lambda-coef', default=0.0, type=float)
parser.add_argument('--local-learning-rate', default=0.1, type=float)
parser.add_argument('--global-learning-rate', default=1.0, type=float)
parser.add_argument('--lr-decay', default=0.9995, type=float)
parser.add_argument('--sch-gamma', default=1.0, type=float)
parser.add_argument('--test-per', default=1, type=int)
parser.add_argument('--batchsize', default=50, type=int)
parser.add_argument('--seed', default=20, type=int)
parser.add_argument('--samlr', default=0.1, type=float)
parser.add_argument('--out-file', default='res/', type=str)
parser.add_argument('--cuda', default=0, type=int)
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    device = torch.device(args.cuda)
else:
    device = torch.device("cpu")

# Dataset initialization
data_path = './'

n_client = args.n_client
# Generate IID or Dirichlet distribution
if args.non_iid is False:
    data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule='iid',
                                 data_path=data_path)
else:
    data_obj = DatasetObject(dataset=args.dataset, n_client=n_client, seed=args.seed, unbalanced_sgm=0, rule=args.rule,
                                 rule_arg=args.rule_arg, data_path=data_path)

''' use util_model_3x3.py to force the first conv layer with 3x3 filter

if args.dataset == 'CIFAR10':
    num_classes = 10
elif args.dataset == 'CIFAR100':
    num_classes = 100
elif args.dataset == 'tinyimagenet':
    num_classes = 200


# use util_model.py to force the first conv layer with 7x7 filter
model_func = lambda : ResNet18(num_classes)
model_name = "resnet18"
'''

# Model function
model_name = args.model
model_func = lambda: client_model(model_name)

# Common hyperparameters
com_amount = args.epochs
save_period = 10000
weight_decay = 1e-3
batch_size = args.batchsize
act_prob = args.act_prob
suffix = model_name
lr_decay_per_round = args.lr_decay
out_file = args.out_file

# judge the file path valid
if not os.path.exists(out_file):
    os.makedirs(out_file)

# Initialize the model for all methods with a random seed or load it from a saved initial model
torch.manual_seed(0)
init_model = model_func()

n = sum(p.numel() for p in init_model.parameters())
print(n)

if __name__=='__main__':
    print('begin to training...')
     
    if args.method == 'MoFedSAM':
        epoch = args.local_epochs
        n_data_per_client = np.concatenate(data_obj.client_x, axis=0).shape[0] / n_client
        n_iter_per_epoch = np.ceil(n_data_per_client / batch_size)
        n_minibatch = (epoch * n_iter_per_epoch).astype(np.int64)

        local_learning_rate = args.local_learning_rate
        global_learning_rate = args.global_learning_rate * local_learning_rate * n_minibatch
        alpha = args.alpha
        test_per = args.test_per
        samlr = args.samlr

        train_MoFedSAM(device=device, data_obj=data_obj, act_prob=act_prob,
            global_learning_rate=global_learning_rate, local_learning_rate=local_learning_rate, alpha=alpha,
            batch_size=batch_size, n_minibatch=n_minibatch,
            com_amount=com_amount, test_per=test_per, weight_decay=weight_decay,
            model_func=model_func, init_model=init_model,
            sch_step=1, sch_gamma=1, rand_seed=0, lr_decay_per_round=lr_decay_per_round, samlr=samlr, out=out_file)


    elif args.method == 'FedAvg':
        epoch = args.local_epochs
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_FedAvg(device=device, data_obj=data_obj,act_prob=act_prob, learning_rate=learning_rate,
                    batch_size=batch_size, epoch=epoch, com_amount=com_amount, test_per=test_per,
                    weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                    sch_step=1, sch_gamma=1, rand_seed=0, lr_decay_per_round=lr_decay_per_round, out=out_file)


    elif args.method == 'FedDyn':
        epoch = args.local_epochs
        alpha_coef = args.alpha_coef
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_FedDyn(device=device, data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                    batch_size=batch_size,  epoch=epoch, com_amount=com_amount, test_per=test_per,
                    weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                    alpha_coef=alpha_coef, sch_step=1, sch_gamma=args.sch_gamma,
                    rand_seed=0, lr_decay_per_round=lr_decay_per_round, out=out_file)
    

    elif args.method == 'FedSMOO':
        epoch = args.local_epochs
        alpha_coef = args.alpha_coef
        learning_rate = args.local_learning_rate
        test_per = args.test_per

        train_FedSMOO(device=device, data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                    batch_size=batch_size,  epoch=epoch, com_amount=com_amount, test_per=test_per,
                    weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                    alpha_coef=alpha_coef, sch_step=1, sch_gamma=args.sch_gamma, samlr=args.samlr,
                    rand_seed=0, lr_decay_per_round=lr_decay_per_round, out=out_file)
    