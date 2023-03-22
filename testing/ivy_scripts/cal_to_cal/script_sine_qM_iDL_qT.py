#!/usr/bin/env python
# encoding: utf-8
"""
@author: ivy
@contact: ivyivyzhao77@gmail.com
@software: PyCharm 2022.3
@file: script_sine_qM_iDL_qT.py
@time: 2023/3/14 11:21
"""

from train._utils import *

time_step = 10
BATCH_SIZE = 128
LR = 0.1
EPOCH = 100
num_worms_train = 1

model = get_model(OmegaConf.load("conf/model.yaml"))
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss(reduction="mean")

dataset = get_dataset(OmegaConf.load("conf/dataset.yaml"))

print(dataset["worm0"].keys())

plt.plot(dataset["worm0"]["calcium_data"][:, 1])
plt.plot(dataset["worm0"]["residual_calcium"][:, 1])
plt.show()

for k in range(num_worms_train):
    worm = "worm" + str(k)
    calcium_data = dataset[worm]["calcium_data"]
    residual_calcium = dataset[worm]["residual_calcium"]

    cutdown = calcium_data.shape[0]
    train_size = int(0.8 * cutdown)
    test_size = cutdown - train_size

    train_ca, test_ca = calcium_data.split(train_size, dim=0)
    train_res, test_res = residual_calcium.split(train_size, dim=0)

    train_loop_cal = []
    train_loop_res = []
    for i in range(0, train_ca.shape[0] - time_step):
        train_loop_res.append(np.array(train_res[i : i + time_step, :]))
        train_loop_cal.append(np.array(train_ca[i : i + time_step, :]))

    train_loop_cal = torch.tensor(np.array(train_loop_cal))
    train_loop_res = torch.tensor(np.array(train_loop_res))

    # print(train_loop_cal.shape, train_loop_res.shape)

    test_loop_cal = []
    test_loop_res = []
    for i in range(0, test_ca.shape[0] - time_step):
        test_loop_res.append(np.array(test_res[i : i + time_step, :]))
        test_loop_cal.append(np.array(test_ca[i : i + time_step, :]))

    test_loop_cal = torch.tensor(np.array(test_loop_cal))
    test_loop_res = torch.tensor(np.array(test_loop_res))
    train_loop_cal_target = torch.zeros_like(train_loop_cal)
    train_loop_cal_target[:-1] = train_loop_cal[1:]

    test_loop_cal_target = torch.zeros_like(test_loop_cal)
    test_loop_cal_target[:-1] = test_loop_cal[1:]

    train_dataset = torch.utils.data.TensorDataset(
        train_loop_cal, train_loop_cal_target
    )
    test_dataset = torch.utils.data.TensorDataset(test_loop_cal, test_loop_cal_target)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("\n")
    print("--------worm " + "No. " + str(k) + "--------")
    # train_loader = worm_train[i]
    # test_loader = worm_test[i]
    train_base_loss_history = []
    train_pred_loss_history = []
    test_base_loss_history = []
    test_pred_loss_history = []
    for e in range(EPOCH):
        train_base_loss = 0
        train_pred_loss = 0
        test_base_loss = 0
        test_pred_loss = 0
        count = 0
        for X_train, Y_train in train_loader:
            mask_train = dataset[worm]["named_neurons_mask"]
            X_train, Y_train = torch.tensor(X_train, requires_grad=True), torch.tensor(
                Y_train, requires_grad=True
            )
            optimizer.zero_grad()
            # Baseline: loss if the model predicted the residual to be 0
            # print(Y_train.shape, mask.shape)
            mask_train = mask_train.repeat(Y_train.shape[1], 1)
            mask_train = mask_train.unsqueeze(0)
            mask_train = mask_train.repeat(Y_train.shape[0], 1, 1)
            # print(Y_train.shape, mask.shape)
            base = criterion(X_train * mask_train, Y_train * mask_train)
            # print(Y_train*mask, (Y_train*mask).shape)
            # print(X_train.shape)
            Y_tr = model(X_train)  # .transpose(1, 2))  # Forward pass.
            Y_tr = Y_tr  # .transpose(1, 2)
            loss = criterion(Y_tr * mask_train, Y_train * mask_train)
            loss.backward(retain_graph=True)
            optimizer.step()  # Update parameters based on gradients.
            train_base_loss += base.detach().item()
            train_pred_loss += loss.detach().item()
            count += 1
        train_base_loss_history.append(train_base_loss / count)
        train_pred_loss_history.append(train_pred_loss / count)
        count = 0
        for X_test, Y_test in test_loader:
            mask_test = dataset[worm]["named_neurons_mask"]
            X_test, Y_test = torch.tensor(X_test, requires_grad=True), torch.tensor(
                Y_test, requires_grad=True
            )
            optimizer.zero_grad()  # Clear gradients.
            mask_test = mask_test.repeat(Y_test.shape[1], 1)
            mask_test = mask_test.unsqueeze(0)
            mask_test = mask_test.repeat(Y_test.shape[0], 1, 1)
            # Baseline: loss if the model predicted the residual to be 0
            base = criterion(X_test * mask_test, Y_test * mask_test)
            # X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1])).transpose(1, 2)
            Y_pred = model(X_test)  # .transpose(1, 2))  # Forward pass.
            Y_pred = Y_pred  # .transpose(1, 2)
            loss = criterion(Y_pred * mask_test, Y_test * mask_test)
            test_base_loss += base.detach().item()
            test_pred_loss += loss.detach().item()
            count += 1
        test_base_loss_history.append(test_base_loss / count)
        test_pred_loss_history.append(test_pred_loss / count)

        print(
            "epoch = {}, train_loss = {:.4f}, test_loss = {:.4f}".format(
                e,
                train_pred_loss_history[-1] - train_base_loss_history[-1],
                test_pred_loss_history[-1] - test_base_loss_history[-1],
            )
        )

    delta_train = [
        train_pred_loss_history[i] - train_base_loss_history[i]
        for i in range(len(train_base_loss_history))
    ]
    delta_test = [
        test_pred_loss_history[i] - test_base_loss_history[i]
        for i in range(len(test_base_loss_history))
    ]
    plt.plot(delta_train)
    plt.plot(delta_test)
    plt.legend(["train", "test"])
    plt.ylabel("Loss-Baseline")
    plt.xlabel("Epoch")
    plt.title("Worm " + str(k) + ", Epoch: " + str(EPOCH) + ", Loss: MSE")
    plt.show()

# visualization of prediction
pred = []
target = []
X_test = []
Y_test = []

num_neurons = 302
for i in range(time_step, calcium_data.shape[0] - 1):
    X, Y = (
        calcium_data[i - time_step : i, :num_neurons],
        calcium_data[i - time_step + 1 : i + 1, :num_neurons],
    )
    X_test.append(np.array(X))
    Y_test.append(np.array(Y))

X_test = torch.tensor(np.array(X_test))
Y_test = torch.tensor(np.array(Y_test))
print(X_test.shape)
optimizer.zero_grad()  # Clear gradients.
mask_test = dataset[worm]["named_neurons_mask"]
mask_test = mask_test.repeat(Y_test.shape[1], 1)
mask_test = mask_test.unsqueeze(0)
mask_test = mask_test.repeat(Y_test.shape[0], 1, 1)
# X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1])).transpose(1, 2)
Y_pred = model(X_test)  # .transpose(1, 2))  # Forward pass.
print(X_test.shape)
Y_pred = Y_pred  # .transpose(1, 2)
print(Y_pred.shape, Y_test.shape)


def draw_pic(Y_pred, Y_test, num, time):
    plt.plot(Y_test.detach().numpy()[:time, 0, num])
    plt.plot(Y_pred.detach().numpy()[:time, 0, num])
    plt.legend(["target", "pred"], loc="upper right")
    plt.title("Plot for Neuron " + str(num) + " from time 0 to " + str(time))
    plt.show()


draw_pic(Y_pred, Y_test, 0, dataset["worm0"]["max_time"])

draw_pic(Y_pred, Y_test, 1, dataset["worm0"]["max_time"])

draw_pic(Y_pred, Y_test, 60, dataset["worm0"]["max_time"])

draw_pic(Y_pred, Y_test, 120, dataset["worm0"]["max_time"])

draw_pic(Y_pred, Y_test, 300, dataset["worm0"]["max_time"])
