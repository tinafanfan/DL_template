import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import IrisDataset
from model import FeedForwardNeuralNet
from train import train

def main():
    # set parameters
    ## training
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch",  type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-1, help="learning rate")
    ## model
    parser.add_argument("--input_size",  type=int, default=4, help="imput size of network")
    parser.add_argument("--hidden_size", type=int, default=4, help="hidden size of network")
    parser.add_argument("--num_classes", type=int, default=3, help="output size of network")
    ## env
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # set gpu
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # load data
    dataset = IrisDataset() 
    """
    還缺train/test, 
    (1) 檔案不同直接寫在 dataset 裏面, https://github.com/tinafanfan/Conditional-seq2seq-VAE/blob/main/utils.py#L52
    (2) 再寫一個切割的函數，並利用 DataLoader 中的 sampler, https://ithelp.ithome.com.tw/articles/10277787
    """
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # assign model
    model = FeedForwardNeuralNet(args).to(device)

    # assign optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # train
    train(args, train_dataloader, model, optimizer, criterion, device)


if __name__ == "__main__":
    main()