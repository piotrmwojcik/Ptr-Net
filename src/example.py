from matplotlib.ticker import MaxNLocator
from torch import optim
from torch.utils.data import DataLoader

from datasets.sort_dataset import SortDataset
from ptr_net.ptr_net import PointerNet
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt

from src import USE_CUDA

if __name__ == "__main__":
    pointer = PointerNet(embedding_size=32, hidden_size=32, seq_len=10, n_glimpses=1, tanh_exploration=10,
                         use_tanh=True)
    adam = optim.Adam(pointer.parameters(), lr=1e-4)

    if USE_CUDA:
        pointer = pointer.cuda()

    train_size = 1000
    val_size = 100
    train_dataset = SortDataset(10, train_size)
    val_dataset = SortDataset(10, val_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    n_epochs = 1
    train_loss = []
    val_loss = []

    print('Before:\n')
    test_inp = Variable(torch.tensor([[3, 2, 1, 4, 7, 0, 9, 6, 5, 8]]))
    _, test = pointer(test_inp, Variable(torch.sort(test_inp)[0]))
    print(test)

    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        for batch_id, sample_batch in enumerate(train_loader):
            inputs = Variable(sample_batch)
            target = Variable(torch.sort(sample_batch)[0])
            if USE_CUDA:
                inputs = inputs.cuda()
                target = target.cuda()

            loss, _ = pointer(inputs, target)
            train_loss.append(loss.item())
            epoch_train_loss += loss

            adam.zero_grad()
            loss.backward()
            adam.step()

            # train_loss.append(epoch_train_loss / len(train_loader))

            # print(f"Epoch {epoch} train loss: {train_loss[-1]}")
            if batch_id % 100 == 0:
                pointer.eval()
                for val_batch_id, val_batch in enumerate(val_loader):
                    inputs = Variable(val_batch)
                    target = Variable(torch.sort(val_batch)[0])
                    if USE_CUDA:
                        inputs = inputs.cuda()
                        target = target.cuda()
                    loss, ret = pointer(inputs, target)
                    val_loss.append(loss.item())
                    epoch_val_loss += loss

        # val_loss.append(epoch_val_loss / len(val_loader))
        # print(f"Epoch {epoch} validation loss: {val_loss[-1]}")

    print('After:\n')
    test_inp = Variable(torch.tensor([[3, 2, 1, 4, 7, 0, 9, 6, 5, 8]]))
    _, test = pointer(test_inp, Variable(torch.sort(test_inp)[0]))
    print(test)

    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()
