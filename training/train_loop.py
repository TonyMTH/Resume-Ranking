import copy
import pickle

import numpy as np
import torch

from training.evaluation import Evaluate

# Used for plotting and display of figures
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, valid_loader, k_rank,
               printing_gap, saved_model_device, model_path, device, PIK_plot_data, scheduler):
    train_losses = []
    test_losses = []
    valid_losses = []
    lest_valid_loss = -np.inf

    for epoch in range(epochs):
        loss_train = 0

        for _, labels, features in train_loader:
            labels, features = labels.float().view(-1, 1).to(device), features.float().to(device)

            output = model.forward(features)  # 1) Forward pass
            loss = criterion(output, labels)  # 2) Compute loss

            optimizer.zero_grad()
            loss.backward()  # 3) Backward pass
            optimizer.step()  # 4) Update model
            loss_train += loss.item()

        model.eval()

        with torch.no_grad():
            train_num_correct = 0
            train_num_samples = 0
            for qid, labels, features in train_loader:
                labels_torch, features = labels.view(-1, 1).float().to(device), features.float().to(device)

                output = model(features)
                # _, predictions = output.max(1)
                pred = output.to('cpu')  # pred = predictions.to('cpu')

                train_num_correct += Evaluate().mean_ndcg(labels, pred, qid, k_rank)
                train_num_samples += pred.size(0)

        with torch.no_grad():
            test_num_correct = 0
            test_num_samples = 0
            for qid, labels, features in test_loader:
                labels_torch, features = labels.view(-1, 1).float().to(device), features.float().to(device)

                output = model(features)
                # _, predictions = output.max(1)
                pred = output.to('cpu')

                test_num_correct += Evaluate().mean_ndcg(labels, pred, qid, k_rank)
                test_num_samples += pred.size(0)

        with torch.no_grad():
            valid_num_correct = 0
            valid_num_samples = 0
            for qid, labels, features in valid_loader:
                labels_torch, features = labels.view(-1, 1).float().to(device), features.float().to(device)

                output = model(features)
                # _, predictions = output.max(1)
                pred = output.to('cpu')

                valid_num_correct += Evaluate().mean_ndcg(labels, pred, qid, k_rank)
                valid_num_samples += pred.size(0)

        train_loss = float(train_num_correct) / train_num_samples
        test_loss = float(test_num_correct) / test_num_samples
        valid_loss = float(valid_num_correct) / valid_num_samples

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        valid_losses.append(valid_loss)

        # Save best model
        # if valid_loss > lest_valid_loss:
        #     lest_valid_loss = valid_loss
        #
        #     best_model_state = copy.deepcopy(model)
        #     best_model_state.to(saved_model_device)
        #     torch.save(best_model_state, model_path)

        if epoch % printing_gap == 0:
            print('Epoch: {}/{}\t.............'.format(epoch, epochs), end=' ')
            print("Loss: {:.4f}".format(loss_train / train_num_samples), end=' ')
            print("Train NDCG: {:.4f}".format(train_loss), end=' ')
            print("Test NDCG: {:.4f}".format(test_loss), end=' ')
            print("Valid NDCG: {:.4f}".format(valid_loss), end=' ')
            print("Best Valid NDCG: {:.4f}".format(lest_valid_loss))

            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(model, model_path)

            # Save data to pickle
            data = {'train_loss': train_loss, 'test_loss': test_loss, 'valid_loss': valid_loss}
            with open(PIK_plot_data, "wb") as f:
                pickle.dump(data, f)

            lr = scheduler.get_last_lr()[0]

        model.train()

    plt.plot(train_losses, label="Train NDCG")
    plt.plot(test_losses, label="Test NDCG")
    plt.plot(valid_losses, label="Valid NDCG")
    plt.xlabel(" Iteration ")
    plt.ylabel("Loss value")
    plt.legend(loc="upper left")
    plt.show()
