import torch;
import torch.nn as nn;
import torch.optim as optim;
from torch.utils.data import DataLoader

import neuronNetwork as nnd;
from dataset import DataSetMode


def train(net, dataSet):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    net = net.to(device);

    dataSet.resetDataSet();

    optimizer = optim.Adam(net.parameters());
    criterion = nn.CrossEntropyLoss();

    for epoch in range(100):
        runningLoss = 0;

        # Training
        dataSet.setMode(DataSetMode.TRAINING);
        dataSetLoader = DataLoader(dataSet, batch_size=4096, shuffle=True);
        for ecg, label, index in dataSetLoader:
            inputs = ecg.to(torch.float32).to(device);
            target = label.to(torch.long).to(device);

            optimizer.zero_grad();

            output = net(inputs);

            loss = criterion(output, target);
            loss.backward();
            optimizer.step();

            runningLoss += loss.item();

            print("Batch Loss = " + str(loss.item()));

        print("Epoch " + str(epoch) + ": Loss = " + str(runningLoss));

        # Positive validation
        dataSet.setMode(DataSetMode.VALIDATION_POSITIVE);
        dataSetLoader = DataLoader(dataSet, batch_size=4096, shuffle=True);
        correctPositive = 0;
        positiveValidationSetSize = 0;
        for ecg, label, index in dataSetLoader:
            inputs = ecg.to(torch.float32).to(device);
            target = label.to(torch.long).to(device);
            output = net(inputs);

            prediction = torch.argmax(output, 1);
            correctPositive += (prediction == target).sum().float();

            positiveValidationSetSize += len(target);

        positiveAccuracy = (correctPositive / positiveValidationSetSize).to("cpu").item();
        print("Positive validation set, accuracy:" + str(positiveAccuracy));

        # Negative validation and hard negative mining
        dataSet.setMode(DataSetMode.VALIDATION_NEGATIVE);
        dataSetLoader = DataLoader(dataSet, batch_size=256, shuffle=True);
        hardNegativeMiningThreshold = .80;
        hardNegativeMiningList = [];
        correctNegative = 0;
        negativeValidationSetSize = 0;
        for ecg, label, index in dataSetLoader:
            inputs = ecg.to(torch.float32).to(device);
            target = label.to(torch.long).to(device);
            output = net(inputs);

            prediction = torch.argmax(output, 1);
            correctNegative += (prediction == target).sum().float();

            negativeValidationSetSize += len(target);

            # Refresh hard negative mining list
            if (correctNegative.to("cpu").item() / len(target)) <= hardNegativeMiningThreshold:
                hardNegativeMiningList += index;

        negativeAccuracy = (correctNegative / negativeValidationSetSize).to("cpu").item();
        print("Negative validation set, accuracy:" + str(negativeAccuracy));
        print("Hard negative items: " + str(len(hardNegativeMiningList)));
        dataSet.hardNegativeMining(hardNegativeMiningList);

        if (positiveAccuracy + negativeAccuracy) / 2 > .995:
            break;

    print("Training finished.");
    torch.save(net.state_dict(), "./parameters.pkl");
    torch.save(net, "./model.pkl");
