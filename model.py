import csv
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from scipy.integrate import quad
from torch.nn.functional import normalize
from pretraitement import alignement, suppression_water,load_dcm_data
from torchvision import datasets, transforms    
"""
class Model(nn.Module):
    def __init__(self, num_output):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            

            nn.MaxPool1d(2), # Je divise mon vecteur de 2 donc de 1024 a 512

            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool1d(2), # vecteur de 256 points

            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
        
            nn.MaxPool1d(2), # vecteur de 128 points

            nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)  # Réduire la taille de sortie à 193

        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),  
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),  

            nn.Conv1d(256, 128, kernel_size=1),  
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'), 

            nn.Conv1d(128, 64, kernel_size=1),  
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'), 

            nn.Conv1d(64, 32, kernel_size=1), 
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv1d(32, 1, kernel_size=7),  # Taille de la sortie
            nn.ReLU()

        )

        # Ajoutez les couches pour prédire les "Largeurs à mi-hauteur" (Width)
        self.amplitude_predictor = nn.Sequential(
            nn.Flatten(), #flatten pour passer en 1 dimmension puis lineaire n, 1 avec n = batch
            nn.Linear( 262144,50),
            nn.ReLU(),
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)  # Assurez-vous que cela correspond au nombre de métabolites
        )

        # Ajoutez les couches pour prédire les "Largeurs à mi-hauteur" (Width)
        self.Width_predictor = nn.Sequential(
            nn.Flatten(), #flatten pour passer en 1 dimmension puis lineaire n, 1 avec n = batch
            nn.Linear( 262144,50),
            nn.ReLU(),
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)  # Assurez-vous que cela correspond au nombre de métabolites
        )

         # Type predictor
        self.type_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(262144, 50),
            nn.ReLU(),
            nn.Linear(50, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output),
            nn.Sigmoid()  #fonction d'activation sigmoïde pour la classification binaire
        )
    def forward(self, spectrum):
        
        conv_out = self.encoder(spectrum)

        reconstructed_spectrum = self.decoder(conv_out)

        amplitude = self.amplitude_predictor(conv_out)

        Width = self.Width_predictor(conv_out)

        type_pred = self.type_predictor(conv_out.flatten(1))

        return amplitude, Width, reconstructed_spectrum, type_pred
"""

class Model(nn.Module):
    def __init__(self, num_output):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.ReLU()
        )

        self.amplitude_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)
        )

        self.width_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output)
        )

        self.type_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 512, 1024),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_output),
            nn.Sigmoid()  
        )

    def forward(self, spectrum):
        conv_out = self.encoder(spectrum)
        reconstructed_spectrum = self.decoder(conv_out)
        amplitude = self.amplitude_predictor(conv_out)
        width = self.width_predictor(conv_out)
        type = self.type_predictor(conv_out)
        return amplitude, width, reconstructed_spectrum, type
