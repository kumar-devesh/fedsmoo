# code to initialize all model params to zeros

for param in model.parameters():
    param.data.fill_(0.0)

# code to perform operations on a set of two models 

for param1, param2 in zip(model.parameters(), model.parameters()):
     param1.data += param2.data
 