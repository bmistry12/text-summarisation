# cross validation

class KFoldsCrossValidation():
    def __init__(self, n_folds):
        self.folds = n_folds
        self.models_set = ""
        self.test_set = ""
        self.validation_set = ""

class LeaveOutOneCrossValidation():
    def __init__(self):
        pass
