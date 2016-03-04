# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train".

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util

TRAIN_DIR = "train"

call_set = set([])
malware_classes = ["Agent", "AutoRun", "FraudLoad", "FraudPack", "Hupigon", "Krap","Lipler",
                           "Magania", "None", "Poison", "Swizzor", "Tdss", "VB", "Virut", "Zbot"]
tagnums =[[] for x in xrange(len(malware_classes))]
def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = []
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        counter = 0
        for node in tree.iter():
            print node.text
    #         counter = counter + 1
    #     tagnums[util.malware_classes.index(clazz)].append(counter)
    # print tagnums


## Feature extraction
def main():
    create_data_matrix(0, 1, TRAIN_DIR)

    # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).

if __name__ == "__main__":
    main()


