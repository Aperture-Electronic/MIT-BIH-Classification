class markedECG:
    def __init__(self, ecgData, ecgClassification):
        self.data = ecgData;
        self.classification = ecgClassification;

    @property
    def getECGData(self): return self.data;

    # Classification of ECG
    # 0: 1(N) 12(/) 14(~) 13(Q) Normal beat
    # 1: 2(L) Left bundle branch block beat
    # 2: 3(R) Right bundle branch block beat
    # 3: 4(a) Aberrated atrial premature beat, 8(A) Atrial premature beat
    # 4: 5(V) Premature ventricular contraction, 9(S) Premature or ectopic supraventricular beat
    # 5: 7(J) Nodal (junctional) premature beat
    # 6: 10(E) 11(j) Escape beats
    # 7: Other beats
    @property
    def Classification(self): return self.classification;


