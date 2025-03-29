import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        raise NotImplementedError("Model class is not implemented")


def graph():
    raise NotImplementedError("Graph function is not implemented")

torch.manual_seed(392)

dataframe = pd.read_csv(r"california-housing-prices-dataset\housing.csv")
