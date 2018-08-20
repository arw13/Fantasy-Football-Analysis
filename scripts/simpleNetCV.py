""" Simple neural net with single season """

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
sns.set()

# load in data to use in training
data_dir = '../sortedSeasonData/'
df_data  = pd.read_csv(data_dir + 'allSeasons.csv')

# create tensors to hold input and outputs
# x = torch.tensor(df_data.iloc[:,2:-2].values)
# y = torch.tensor(df_data.iloc[:,-1].values)
x = torch.tensor(df_data.iloc[:,1:].values)


# format x as a tensorfloat
x = x.float()
# y = y.float()
# y = y.view(len(x),1)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, x.shape[1]-1, 100 , 1


# Define the indices
indices = list(range(len(x))) # start with all the indices in training set
split = 1000 # define the split size

# Define your batch_size
batch_size = 64

# Random, non-contiguous split
validation_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(validation_idx))

# Contiguous split
# train_idx, validation_idx = indices[split:], indices[:split]


# define our samplers -- we use a SubsetRandomSampler because it will return
# a random subset of the split defined by the given indices without replacement
train_sampler = SubsetRandomSampler(train_idx)
validation_sampler = SubsetRandomSampler(validation_idx)


# Create the train_loader -- use your real batch_size which you
# I hope have defined somewhere above
train_loader = torch.utils.data.DataLoader(dataset=x,
                batch_size=batch_size, sampler=train_sampler)

# You can use your above batch_size or just set it to 1 here.  Your validation
# operations shouldn't be computationally intensive or require batching.
validation_loader = torch.utils.data.DataLoader(dataset=x,
                batch_size=1, sampler=validation_sampler)



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

loss_storage = []
lossCV_storage = []
max_epochs = 20

for epoch in range(max_epochs):
    for local_batch in train_loader:
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(local_batch[:,:-1])

        y = local_batch[:,-1]
        y = y.float()
        y = y.view(-1,1)
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


        # Validation process
        for local_batch in validation_loader:
            # Your model
             # Forward pass: compute predicted y by passing local_batch to the model.
            y_pred = model(local_batch[:,:-1])

            y = local_batch[:,-1]
            y = y.float()
            y = y.view(-1,1)
            # Compute and print loss.
            loss = loss_fn(y_pred, y)

            # print(t, loss.item())
            lossCV_storage.append(loss)


            # progressbar
    print("Progress {:2.1%}".format(epoch / max_epochs), end="\r")



max_epochs = 25

plt.hold()
plt.plot(range(max_epochs), loss_storage)
plt.plot(range(max_epochs), lossCV_storage)
plt.legend('Training Loss', 'CV Loss')
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
