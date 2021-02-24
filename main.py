from torch.utils.data import DataLoader

import training
from dataset import MITBIHDataSet;
from neuronNetwork import cnn;

dataSetPath = "D:/Projects/CSharp/MIT-BIH-Marker/MITBIHMarker/MITBIHMarker/bin/Debug/netcoreapp3.1/datamark/";
dataSetSubSet = "MLII";

net = cnn();
dataSet = MITBIHDataSet(dataSetPath, dataSetSubSet);

training.train(net, dataSet);
