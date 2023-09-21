from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_optimizer import *
from .utils import *
# Global parameters

import time
max_norm = 10

# --- Training models

def train_model(device, model, train_x, train_y, test_x, test_y, learning_rate, batch_size, epoch, print_per, weight_decay, dataset_name, sch_step=1, sch_gamma=1, print_verbose=False):
    n_train = train_x.shape[0]
    train_gen = data.DataLoader(Dataset(train_x, train_y, train=True, dataset_name=dataset_name), batch_size=batch_size, shuffle=True)                          
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put test_x=False if no test data given
    print_test = not isinstance(test_x, bool)
    
    model.train()
    
    for e in range(epoch):
        # Training
        train_gen_iter = train_gen.__iter__()
        for i in range(int(np.ceil(n_train/batch_size))):

            batch_x, batch_y = train_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()

        if print_verbose and (e == 0 or (e+1) % print_per) == 0:
            loss_train, acc_train = get_acc_loss(device, train_x, train_y, model, dataset_name, weight_decay)
            if print_test:
                loss_test, acc_test = get_acc_loss(device, test_x, test_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f" 
                      %(e+1, acc_train, loss_train, acc_test, loss_test, scheduler.get_lr()[0]))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" %(e+1, acc_train, loss_train, scheduler.get_lr()[0]))

            model.train()
        scheduler.step()
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

