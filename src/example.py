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

    n_epochs = 100
    train_loss = []
    val_loss = []

    for epoch in range(n_epochs):
        for batch_id, sample_batch in enumerate(train_loader):

            inputs = Variable(sample_batch)
            target = Variable(torch.sort(sample_batch)[0])
            if USE_CUDA:
                inputs = inputs.cuda()
                target = target.cuda()

            loss = pointer(inputs, target)

            adam.zero_grad()
            loss.backward()
            adam.step()

            train_loss.append(loss.item())

            if batch_id % 10 == 0:
                #clear_output(True)
                #plt.figure(figsize=(20, 5))
                #plt.subplot(131)
                #plt.title('train epoch %s loss %s' % (epoch, train_loss[-1] if len(train_loss) else 'collecting'))
                #plt.plot(train_loss)
                #plt.grid()
                #plt.subplot(132)
                #plt.title('val epoch %s loss %s' % (epoch, val_loss[-1] if len(val_loss) else 'collecting'))
                #plt.plot(val_loss)
                #plt.grid()
                #plt.show()

                print('train epoch %s loss %s' % (epoch, train_loss[-1] if len(train_loss) else 'collecting'))
                print('val epoch %s loss %s' % (epoch, val_loss[-1] if len(val_loss) else 'collecting'))

            if batch_id % 100 == 0:
                pointer.eval()
                for val_batch in val_loader:
                    inputs = Variable(val_batch)
                    target = Variable(torch.sort(val_batch)[0])
                    if USE_CUDA:
                        inputs = inputs.cuda()
                        target = target.cuda()

                    loss = pointer(inputs, target)
                    val_loss.append(loss.item())
