from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_optimizer import *
from .utils import *
# Global parameters

import time
max_norm = 10

# --- Training models

def train_model_GS(device, model, model_func, alpha_coef, avg_mdl_param, hist_params_diff, hist_sam_diff_list, gs_diff_curr_list,
                    train_x, train_y, learning_rate, batch_size, epoch, print_per,
                    weight_decay, dataset_name, sch_step, sch_gamma, samlr, print_verbose=False):
    
    n_train = train_x.shape[0]

    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=alpha_coef+weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer, step_size=sch_step, gamma=sch_gamma)
    
    model.train()
    model = model.to(device)
    optimizer = GSAM(model.parameters(), base_optimizer, rho=samlr, beta=1.0, gamma=1.0, adaptive=False,
                     nograd_cutoff=0.05)
    
    model.train()

    for e in range(epoch):
        # Training
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train/batch_size))):
            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_y = batch_y.reshape(-1).long()
            
            def defined_backward(loss):
                loss.backward()
            paras = [batch_x, batch_y, loss_fn, model, defined_backward]
            optimizer.paras = paras
            hist_sam_diff_list = optimizer.step(hist_sam_diff_list, gs_diff_curr_list)
            
            ## Get f_i estimate 
            # loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())  
            # loss_f_i = loss_f_i / list(batch_y.size())[0]
            
            # Get linear penalty on the current parameter estimates
            local_par_list = None
            for param in model.parameters():
                if not isinstance(local_par_list, torch.Tensor):
                # Initially nothing to concatenate
                    local_par_list = param.reshape(-1)
                else:
                    local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
            loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            
            loss = loss_algo

            ###
            # base_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            base_optimizer.step()

        if print_verbose and (e+1) % print_per == 0:
            loss_train, acc_train = get_acc_loss(device, train_x, train_y, model, dataset_name, weight_decay)
            print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
            e + 1, acc_train, loss_train, scheduler.get_lr()[0]))
            
        model.train()
        scheduler.step()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model, get_H_param_array(hist_sam_diff_list)
