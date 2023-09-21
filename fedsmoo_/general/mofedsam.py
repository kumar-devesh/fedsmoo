from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_optimizer import *
from .utils import *
# Global parameters

import time
max_norm = 10

# --- Training models

def train_MoFedCM_mdl(device, model, model_func, alpha, delta, train_x, train_y,
                       learning_rate, batch_size, n_minibatch, print_per,
                       weight_decay, dataset_name, sch_step, sch_gamma, samlr, print_verbose=False):
    n_train = train_x.shape[0]

    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=sch_step, gamma=sch_gamma)
    model.train()

    optimizer = ESAM(model.parameters(), base_optimizer, rho=samlr, beta=1.0, gamma=1.0, adaptive=False,
                     nograd_cutoff=0.05)
    
    model.train()

    n_par = get_mdl_params([model_func()]).shape[1]

    n_iter_per_epoch = int(np.ceil(n_train / batch_size))
    epoch = np.ceil(n_minibatch / n_iter_per_epoch).astype(np.int64)

    count_step = 0

    step_loss = 0
    n_data_step = 0
    for e in range(epoch):

        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train / batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.reshape(-1).long()

            def defined_backward(loss):
                loss.backward()
            paras = [batch_x, batch_y, loss_fn, model, defined_backward]
            optimizer.paras = paras
            optimizer.step(alpha=alpha)

            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

            loss_algo = torch.sum(local_par_list * delta)

            loss = (1-alpha) * loss_algo

            # optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            base_optimizer.step()
            step_loss += loss.item() * list(batch_y.size())[0];
            n_data_step += list(batch_y.size())[0]

        if print_verbose and (count_step) % print_per == 0:
            step_loss /= n_data_step
            if weight_decay != None:
                # Add L2 loss to complete f_i
                params = get_mdl_params([model], n_par)
                step_loss += (weight_decay) / 2 * np.sum(params * params)

            print("Step %3d, Training Loss: %.4f, LR: %.5f"
                  % (count_step, step_loss, scheduler.get_lr()[0]))
            step_loss = 0
            n_data_step = 0

        model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

