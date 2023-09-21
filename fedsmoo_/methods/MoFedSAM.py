from utils_libs import *
from utils_dataset import *
from utils_models import *
from general.mofedsam import *
from general.utils import *

def train_MoFedSAM(device, data_obj, act_prob, global_learning_rate, local_learning_rate, alpha, batch_size, n_minibatch,
                com_amount, test_per, weight_decay, model_func, init_model,
                sch_step, sch_gamma, rand_seed=0, lr_decay_per_round=1, samlr=0.1, out='res/'):
    
    print('## use MoFedSAM Method ##')

    n_client = data_obj.n_client
    client_x = data_obj.client_x
    client_y = data_obj.client_y

    cent_x = np.concatenate(client_x, axis=0)
    cent_y = np.concatenate(client_y, axis=0)

    train_perf = np.zeros((com_amount, 2))
    test_perf = np.zeros((com_amount, 2))
    x_distance_average = np.zeros((com_amount, 1))

    n_par = len(get_mdl_params([model_func()])[0])
    print('## total {:d} parameters ##'.format(n_par))
    delta = np.zeros(n_par).astype('float32')

    client_models = list(range(n_client))
    server_model = model_func().to(device)
    server_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
    
    client_param_list = np.zeros((n_client, n_par)).astype('float32')

    print('## start Federated Framework ##')

    for i in range(com_amount):
        inc_seed = 0
        while (True):
            np.random.seed(i + rand_seed + inc_seed)
            act_list = np.random.uniform(size=n_client)
            act_clients = act_list <= act_prob
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                break

        print('Communication Round', i + 1, flush = True)
        print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

        del client_models

        client_models = list(range(n_client))
        delta_sum = np.zeros(n_par).astype('float32')
        prev_params = get_mdl_params([server_model], n_par)[0]

        learning_rate = local_learning_rate * (lr_decay_per_round ** i)

        for client in selected_clients:
            train_x = client_x[client]
            train_y = client_y[client]

            client_models[client] = model_func().to(device)

            client_models[client].load_state_dict(copy.deepcopy(dict(server_model.named_parameters())))

            for params in client_models[client].parameters():
                params.requires_grad = True

            client_models[client] = train_MoFedCM_mdl(device, client_models[client], model_func, alpha,
                                                        torch.tensor(delta, dtype=torch.float32, device=device),
                                                        train_x, train_y, learning_rate, batch_size, n_minibatch,
                                                        5, weight_decay, data_obj.dataset, sch_step,
                                                        sch_gamma, samlr)
            curr_model_param = get_mdl_params([client_models[client]], n_par)[0]
            client_param_list[client] = curr_model_param
            delta_sum += curr_model_param - prev_params

        delta = -delta_sum / (len(selected_clients) * n_minibatch * learning_rate)

        server_model_params = prev_params - global_learning_rate * (lr_decay_per_round ** i) * delta
        
        for client in selected_clients:
            x_distance_average[i] += np.linalg.norm(client_param_list[client] - server_model_params)/len(selected_clients)
        
        server_model = set_client_from_params(device, model_func().to(device), server_model_params)

        if (i + 1) % test_per == 0:
            loss_test, acc_test = get_acc_loss(device, data_obj.test_x, data_obj.test_y,
                                               server_model, data_obj.dataset, 0)
            test_perf[i] = [loss_test, acc_test]

            print("**** Communication %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_test, loss_test), flush = True)

            loss_train, acc_train = get_acc_loss(device, cent_x, cent_y,
                                             server_model, data_obj.dataset, 0)
            train_perf[i] = [loss_train, acc_train]
            print("**** Communication %3d, Training Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_train, loss_train), flush = True)

        # Freeze model
        for params in server_model.parameters():
            params.requires_grad = False
    
    return
