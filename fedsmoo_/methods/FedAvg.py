from utils_libs import *
from utils_dataset import *
from utils_models import *
from general.fedavg import *
from general.utils import *


def train_FedAvg(device, data_obj, act_prob ,learning_rate, batch_size,
                 epoch, com_amount, test_per, weight_decay,
                 model_func, init_model, sch_step, sch_gamma,
                 rand_seed=0, lr_decay_per_round=1, out='res/'):
    
    print('## use FedAvg Method ##')
    n_client=data_obj.n_client

    client_x = data_obj.client_x; client_y=data_obj.client_y
    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)
    
    train_perf_sel = np.zeros((com_amount, 2)); train_perf_all = np.zeros((com_amount, 2))
    test_perf_sel = np.zeros((com_amount, 2)); test_perf_all = np.zeros((com_amount, 2))
    n_par = len(get_mdl_params([model_func()])[0])
    print('## total {:d} parameters ##'.format(n_par))

    init_par_list=get_mdl_params([init_model], n_par)[0]
    client_params_list=np.ones(n_client).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_client X n_par
    x_distance_average = np.zeros((com_amount, 1))
    client_models = list(range(n_client))
    avg_model = model_func().to(device)
    avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))

    all_model = model_func().to(device)
    all_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
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

        del client_models
        client_models = list(range(n_client))
        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]
            test_x = False
            test_y = False

            client_models[client] = model_func().to(device)
            client_models[client].load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

            for params in client_models[client].parameters():
                params.requires_grad = True
            client_models[client] = train_model(device, client_models[client], train_x, train_y,
                                            test_x, test_y,
                                            learning_rate * (lr_decay_per_round ** i), batch_size, epoch, 5,
                                            weight_decay,
                                            data_obj.dataset, sch_step, sch_gamma)

            client_params_list[client] = get_mdl_params([client_models[client]], n_par)[0]
        avg_model_param = np.mean(client_params_list[selected_clients], axis=0)
        avg_model = set_client_from_params(device, model_func(), np.mean(client_params_list[selected_clients], axis = 0))

        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(device, data_obj.test_x, data_obj.test_y,
                                             avg_model, data_obj.dataset, 0)
            test_perf_sel[i] = [loss_test, acc_test]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test), flush = True)

            loss_test, acc_test = get_acc_loss(device, cent_x, cent_y,
                                             avg_model, data_obj.dataset, 0)
            train_perf_sel[i] = [loss_test, acc_test]
            print("**** Communication sel %3d, Train Accuracy: %.4f, Loss: %.4f"
                  %(i+1, acc_test, loss_test), flush = True)

        # Freeze model
        for params in avg_model.parameters():
            params.requires_grad = False

    return

