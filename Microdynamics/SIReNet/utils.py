import numpy as np

class IndexSampler(object):

    def __init__(self, indexes):
        self.indexes = indexes

    def sample(self, n_samples):

        sample = np.random.choice(self.indexes,n_samples,replace=False)
        self.indexes = np.array([i for i in self.indexes if i not in sample])
        print(len(self.indexes))

        return sample

    def sample_remaining(self):

        return self.indexes


class IndexSampler2(object):

    def __init__(self, indexes):
        self.indexes = indexes

    def sample(self, test_size, val_size, random_state=42):

        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(self.indexes, test_size=test_size, random_state=random_state)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size)

        return train_idx,val_idx,test_idx



class MaskSampler(object):

    def __init__(self, indexes):
        self.indexes = indexes
        self.size = len(indexes)

    def sample(self, test_size,val_size,random_state=42):

        from sklearn.model_selection import train_test_split

        train_mask = np.zeros(self.size,dtype=np.bool)
        validation_mask = np.zeros(self.size, dtype=np.bool)
        test_mask = np.zeros(self.size, dtype=np.bool)

        train_idx, test_idx = train_test_split(self.indexes,test_size=test_size,random_state=random_state)

        for i in test_idx:
            test_mask[i] = True

        train_idx, val_idx = train_test_split(train_idx,test_size=val_size)

        for i in val_idx:
            validation_mask[i] = True

        for i in train_idx:
            train_mask[i] = True

        return train_mask,validation_mask,test_mask






