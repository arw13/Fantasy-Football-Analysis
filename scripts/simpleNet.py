""" Simple neural net with single season """

import torch
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data.sampler import SubsetRandomSampler
sns.set()

# load in data to use in training
data_dir = '../sortedSeasonData/'
df_data  = pd.read_csv(data_dir + 'allSeasons.csv')

# create tensors to hold input and outputs
x = torch.tensor(df_data.iloc[:,2:-2].values)
y = torch.tensor(df_data.iloc[:,-1].values)

# format x as a tensorfloat
x = x.float()
y = y.float()
y = y.view(len(x),1)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, x.shape[1], 100 , 1

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n = 500
loss_storage = []
for t in range(n):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    # print(t, loss.item())
    loss_storage.append(loss)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    # progressbar
    print("Progress {:2.1%}".format(t / n), end="\r")
    
plt.plot(range(n), loss_storage)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()