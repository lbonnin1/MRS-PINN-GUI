import torch.nn as nn
import torch.optim as optim
from model import Model
from torchvision import datasets, transforms  
from trt import traitement


#loss function for PINN
criterion_amplitude = nn.MSELoss()  # Loss for amplitude prediction
criterion_Width = nn.MSELoss()      # Loss for width prediction
criterion_spectrum = nn.MSELoss()   # Loss for spectrum reconstruction
criterion_fit = nn.MSELoss()        # Loss for fit prediction
criterion_type = nn.MSELoss()      # Loss for type classification


#hyperparameters
type_Facteur= 0.2222
amplitude_Facteur= 0.5
Width_Facteur= 0.2778
spectrum_Facteur =1

# Training configuration
epochs = 300    # Number of training epochs
batch = 50            # Batch size
traitement_instance = traitement()

# Créer un pipeline de transformations en utilisant transforms.Compose
transform = transforms.Compose([transforms.Lambda(traitement_instance)])
# Paths to training and testing data directories
dossier_train = r""
dossier_test = r""

# Dataset split configuration
pourcentage_train = 0.8  # Percentage of data used for training

# Model input/output configuration
input_size = 8192       # Size of the input 
num_output = 13         # Number of metabolite parameters to predict
output_size = 8192      # Size of the reconstructed spectrum output

