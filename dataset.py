import os;
import random;
import torch;
import torch.nn as nn;
import torch.nn.functional as nnfunc;
import numpy as np;

from enum import Enum;
from torch.utils import data;
from scipy import signal as sig;


def readDataSetPaths(path, target):
    # Get all sub-directories in path
    subDir = [];
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            if target in directory: subDir.append(os.path.join(root, directory));

    # Get all data files in sub-directory
    dataSet = [];
    for subdir in subDir:
        for root, dirs, files in os.walk(subdir):
            for file in files:
                isDataFile = file.endswith(".csv");

                if isDataFile:
                    dataSet.append(os.path.join(root, file));

    return dataSet;

def convertECGClass(cls):
    # 0: 1(N) 12(/) 14(~) 13(Q) Normal beat
    # 1: 2(L) Left bundle branch block beat
    # 2: 3(R) Right bundle branch block beat
    # 3: 4(a) Aberrated atrial premature beat, 8(A) Atrial premature beat
    # 4: 5(V) Premature ventricular contraction, 9(S) Premature or ectopic supraventricular beat
    # 5: 7(J) Nodal (junctional) premature beat
    # 6: 10(E) 11(j) Escape beats
    # 7: Other beats
    classReconfigure = {1: 0, 12: 0, 14: 0, 13: 0,
                        2: 1, 3: 2, 4: 3, 8: 3,
                        5: 4, 9: 4, 7: 5, 10: 6,
                        11: 6};

    return classReconfigure.get(cls, 7);

def splitDataSet(dataSetPath):
    positiveDataSet = [];
    negativeDataSet = [];

    for section in dataSetPath:
        markFile = os.path.splitext(section)[0] + ".txt";

        # Read mark file
        f = open(markFile, "r");
        ecgMark = f.readline();
        f.close();

        # Convert classification
        ecgMark = convertECGClass(int(ecgMark));

        # Positive/Negative classification
        if ecgMark == 0: negativeDataSet.append(section);
        else: positiveDataSet.append(section);

    return positiveDataSet, negativeDataSet;

def splitDataSetRandom(dataSet, trainCount):
    trainingSet = random.sample(dataSet, trainCount);
    validationSet = list(set(dataSet) ^ set(trainingSet));
    return trainingSet, validationSet;

class DataSetMode(Enum):
    TRAINING = 0
    VALIDATION_POSITIVE = 1
    VALIDATION_NEGATIVE = 2

class MITBIHDataSet(data.Dataset):
    def __init__(self, rootPath, subSet):
        super(MITBIHDataSet, self).__init__();

        # Get all dataset
        self.dataSetPath = readDataSetPaths(rootPath, subSet);

        # Split data set
        self.positiveDataSet, self.negativeDataSet = splitDataSet(self.dataSetPath);

        # Initialize properties
        self.trainDataSet = [];
        self.validationPositiveDataSet = [];
        self.validationNegativeDataSet = [];
        self.dataSetMode = DataSetMode.TRAINING;

    def resetDataSet(self):
        positiveSetSize = len(self.positiveDataSet);

        # Get sample size
        # Positive : Negative = 3 : 1
        # Train : Validation = 3 : 1
        positiveTrainSetSize = int(positiveSetSize * 3 / 4);
        initialNegativeTrainSetSize = int(positiveSetSize / 3);

        # Split data set randomly with sample size
        trainPositiveSet, validPositiveSet = splitDataSetRandom(self.positiveDataSet, positiveTrainSetSize);
        trainNegativeSet, validNegativeSet = splitDataSetRandom(self.negativeDataSet, initialNegativeTrainSetSize);

        # Set the set
        self.trainDataSet = trainPositiveSet + trainNegativeSet;
        self.validationPositiveDataSet = validPositiveSet;
        self.validationNegativeDataSet = validNegativeSet;

    def setMode(self, mode):
        self.dataSetMode = mode;

    def hardNegativeMining(self, hnmList):
        for index in hnmList:
            item = self.validationNegativeDataSet[index];
            self.trainDataSet.append(item);
            self.validationNegativeDataSet.remove(item);

    def __getitem__(self, item):
        # Get path
        if self.dataSetMode == DataSetMode.TRAINING:
            path = self.trainDataSet[item];
        elif self.dataSetMode == DataSetMode.VALIDATION_POSITIVE:
            path = self.positiveDataSet[item];
        else: # VALIDATION_NEGATIVE
            path = self.negativeDataSet[item];

        markFile = os.path.splitext(path)[0] + ".txt";

        # Read data file
        f = open(path, "r");
        ecgData = f.readlines();
        f.close();

        # Read mark file
        f = open(markFile, "r");
        ecgMark = f.readline();
        f.close();

        # Pretreatment
        #  Resampling
        ecgResample = [sig.resample(ecgData, 256)];
        #  Convert
        ecgMark = convertECGClass(int(ecgMark));

        #  Converting classification
        return torch.tensor(ecgResample), torch.tensor(ecgMark), item;

    def __len__(self):
        if self.dataSetMode == DataSetMode.TRAINING:
            return len(self.trainDataSet);
        elif self.dataSetMode == DataSetMode.VALIDATION_POSITIVE:
            return len(self.positiveDataSet);
        else:  # VALIDATION_NEGATIVE
            return len(self.negativeDataSet);


