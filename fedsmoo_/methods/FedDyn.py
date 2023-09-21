from utils_libs import *
from utils_dataset import *
from utils_models import *
from general.feddyn import *
from general.utils import *


def train_FedDyn(device, data_obj, act_prob,
                  learning_rate, batch_size, epoch, com_amount, test_per,
                  weight_decay, model_func, init_model, alpha_coef,
                  sch_step, sch_gamma, rand_seed, lr_decay_per_round, out='res/'):
    print('## use FedDyn Method ##')
    
    n_client = data_obj.n_client
    client_x = data_obj.client_x; client_y=data_obj.client_y
    
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)
    
    weight_list = np.asarray([len(client_y[i]) for i in range(n_client)])
    weight_list = weight_list / np.sum(weight_list) * n_client
    
    # train_all_clt_perf = np.zeros((com_amount, 2))
    # test_all_clt_perf = np.zeros((com_amount, 2))
    
    train_cur_cld_perf = np.zeros((com_amount, 2))
    test_cur_cld_perf = np.zeros((com_amount, 2))
    
    n_par = len(get_mdl_params([model_func()])[0])
    x_distance_average = np.zeros((com_amount, 1))
    
    hist_params_diffs = np.zeros((n_client, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0]
    client_params_list  = np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_client X n_par
    client_models = list(range(n_client))

    cur_cld_model = model_func().to(device)
    cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

    print('## start Federated Framework ##')

    for i in range(com_amount):
        inc_seed = 0
        while(True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list    = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                break

        print('Communication Round', i + 1, flush = True)
        print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
        cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)

        del client_models
        client_models = list(range(n_client))

        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]

            client_models[client] = model_func().to(device)

            model = client_models[client]
            # Warm start from current avg model
            model.load_state_dict(copy.deepcopy(dict(cur_cld_model.named_parameters())))
            for params in model.parameters():
                params.requires_grad = True
            # Scale down
            alpha_coef_adpt = alpha_coef / weight_list[client] # adaptive alpha coef
            hist_params_diffs_curr = torch.tensor(hist_params_diffs[client], dtype=torch.float32, device=device)
            client_models[client] = train_model_alg(device, model, model_func, alpha_coef_adpt,
                                                 cld_mdl_param_tensor, hist_params_diffs_curr,
                                                 train_x, train_y, learning_rate * (lr_decay_per_round ** i),
                                                 batch_size, epoch, 5, weight_decay,
                                                 data_obj.dataset, sch_step, sch_gamma, print_verbose=False)
            curr_model_par = get_mdl_params([client_models[client]], n_par)[0]
            # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
            hist_params_diffs[client] += curr_model_par-cld_mdl_param
            client_params_list[client] = curr_model_par

        avg_mdl_param_sel = np.mean(client_params_list[selected_clients], axis = 0)

        cld_mdl_param = avg_mdl_param_sel + np.mean(hist_params_diffs, axis=0)
        
        for client in selected_clients:
            x_distance_average[i] += np.linalg.norm(client_params_list[client] - cld_mdl_param)/len(selected_clients)

        # all_model     = set_client_from_params(device, model_func(), np.mean(client_params_list, axis = 0))
        cur_cld_model = set_client_from_params(device, model_func().to(device), cld_mdl_param)

        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(device, cent_x, cent_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test), flush = True)
            train_cur_cld_perf[i] = [loss_test, acc_test]
           
            loss_test, acc_test = get_acc_loss(device, data_obj.test_x, data_obj.test_y,
                                             cur_cld_model, data_obj.dataset, 0)
            print("**** Cur cld Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test))
            test_cur_cld_perf[i] = [loss_test, acc_test]

    return
