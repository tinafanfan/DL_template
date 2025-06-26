import torch
from tqdm import tqdm

def train(args, train_loader, model, optimizer, criterion, device):

    total_iteration = args.num_epoch * len(train_loader)
    for epoch in range(args.num_epoch):
        # v1: use enumerate
        # for i, (datas, labels) in enumerate(train_loader):

        # v1: use tqdm, with + for + set_postfix
        with tqdm(train_loader, unit="batch") as tepoch:
            for datas, labels in tepoch:
                
                datas = datas.to(device)
                labels = labels.to(device)

                # init optimizer
                optimizer.zero_grad()
                
                # forward -> backward -> update
                outputs = model(datas)
                # print(outputs.dtype)
                # print(labels.dtype)
                labels = labels.to(torch.int64)

                loss = criterion(outputs, labels)
                
                loss.backward()

                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

        print(f'epoch {epoch+1}/{args.num_epoch}, loss = {loss.item():.4f}')