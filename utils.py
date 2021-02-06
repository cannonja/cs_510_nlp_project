from torchtext import data
from torchtext.datasets import IMDB, SST


def load_imdb_nltk():
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    
    print("Getting splits")
    train, test = IMDB.splits(TEXT, LABEL)
    
    print("Loading splits")
    train_text = []
    train_label = []
    test_text = []
    test_label = []    
    for tr, ts in zip(train, test):
        train_text.append(' '.join(tr.text))
        train_label.append(tr.label)
        test_text.append(' '.join(ts.text))
        test_label.append(ts.label)
        
    return train_text, train_label, test_text, test_label


def load_sst_nltk():
    TEXT = data.Field()
    LABEL = data.Field(sequential=False)
    
    print("Getting splits")
    train, val, test = SST.splits(TEXT, LABEL)
    
    print("Loading splits")
    train_text = []
    train_label = []
    test_text = []
    test_label = []    
    for tr, vl, ts in zip(train, val, test):
        train_text.extend([' '.join(tr.text), ' '.join(vl.text)])
        train_label.extend([tr.label, vl.label])
        test_text.append(' '.join(ts.text))
        test_label.append(ts.label)
        
    return train_text, train_label, test_text, test_label