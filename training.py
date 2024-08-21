import logging
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import random_split
import json
import torch.nn.functional as F
import numpy as np
import json
import os
import pandas as pd
import seaborn as sns
from pretraitement import load_dcm_data
from model import Model
from config import dossier_test,dossier_train ,transform,batch,pourcentage_train,num_output,epochs,criterion_amplitude,criterion_fit,criterion_spectrum,criterion_type,criterion_Width,output_size,amplitude_Facteur,Width_Facteur,type_Facteur,spectrum_Facteur
from metrics import gaussian_area,lorentzian_area, mean_absolute_percentage_error,r_squared,mse


def calculate_accuracy(predicted_labels, targets):
    predicted_labels = torch.tensor(predicted_labels).detach().cpu().numpy()
    targets = torch.tensor(targets).detach().cpu().numpy()
    correct = (predicted_labels == targets).sum().item()
    total = len(targets)
    accuracy = correct / total
    return accuracy

class MonDataset(Dataset):
    def __init__(self, dossier_base):
        self.dossier_base = dossier_base
        self.liste_patients = os.listdir(dossier_base+'/dcm/')
        self.type_mapping=type_mapping
        self.transform=transform

    def __len__(self):
        return len(self.liste_patients)

    def __getitem__(self, idx):
        patient = self.liste_patients[idx]
        chemin_dossier_patient = os.path.join(self.dossier_base+'/dcm', patient)
        #print(chemin_dossier_patient)
        input_spectre = load_dcm_data(chemin_dossier_patient)
        input_spectre = self.transform(input_spectre)
        input_spectre = torch.tensor(input_spectre, dtype=torch.float32)

        # Récupération des paramètres des métabolites dans leurs fichiers JSON respectifs
        fileObject = open(os.path.join(self.dossier_base+'/json',os.path.splitext(patient)[0] + '.json'), "r")
        path=self.dossier_base+'/json',os.path.splitext(patient)[0] 
        jsonContent = fileObject.read()
        obj_python = json.loads(jsonContent)
        metabolites = [obj_python[i] for i in range(13)]

        amplitudes = [torch.tensor(metabolite.get('Amplitude', 0.0), dtype=torch.float32) for metabolite in metabolites]
        widths = [torch.tensor(metabolite.get('Width', 0.0), dtype=torch.float32) for metabolite in metabolites]
        centers = [torch.tensor(metabolite.get('Center', 0.0), dtype=torch.float32) for metabolite in metabolites]
        types = [torch.tensor(self.type_mapping[metabolite.get('Type', 0.0)], dtype=torch.float32) for metabolite in metabolites]


        return input_spectre,amplitudes,widths,centers,types

class Test_data(Dataset):
    def __init__(self, dossier_base):
        self.dossier_base = dossier_base
        self.liste_patients = os.listdir(dossier_base)
        self.transform=transform
        
    def __len__(self):
        return len(self.liste_patients)
    def __getitem__(self, idx):
      
        patient = self.liste_patients[idx]
        chemin_dossier_patient = os.path.join(self.dossier_base, patient)
        path=self.dossier_base+'/json',os.path.splitext(patient)[0] 
        input_spectre =load_dcm_data(chemin_dossier_patient)
        
        input_spectre = self.transform(input_spectre)
        input_spectre = torch.tensor(input_spectre, dtype=torch.float32)
            
        return input_spectre


type_mapping = {
    "Lorentzienne": 0,
    "Gaussienne": 1,
}
train=True
dataset =  MonDataset(dossier_train)
loader = DataLoader(dataset, batch_size=batch, shuffle=False)
dataset_test =Test_data(dossier_test)
loader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

# Split ----------------------------------------------------------------------------------------------
taille_dataset = len(dataset)  # Total size of the dataset
taille_train = int(pourcentage_train * taille_dataset)  # Size of the training set
taille_val = taille_dataset - taille_train  # Size of the validation set
dataset_train, dataset_val = random_split(dataset, [taille_train, taille_val])

# Loader ----------------------------------------------------------------------------------------------

loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch, shuffle=False)

# Modèle----------------------------------------------------------------------------------------------

# Créez une instance du modèle ConvPINN
model = Model(13)
# Optimizer (e.g., Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel',cooldown=0, min_lr=0, eps=1e-08)



# Boucle d'entraînement ----------------------------------------------------------------------------------------------

train_losses = []  # List to store training losses for each epoch
val_losses = []    # List to store validation losses for each epoch

val_losses_amplitude=[]
val_losses_width=[]
val_losses_type=[]
val_losses_spectrum=[]

train_losses_amplitude=[]
train_losses_type=[]
train_losses_spectrum=[]
train_losses_width=[]

r2_scores_all=[]
mape_all=[]
val_mape_all=[]
mean_r2=[]
val_r2_scores_all=[]
val_mean_r2=[]
mse_all=[]
val_mse_all=[]
lr_history=[]

metabolites = ["PCr + Cr 2","GSH + Glu + Gln","MIns 2","MIns 1","PCh + GPC","PCr + Cr 1","Asp","NAA 2","Gln","Glu","NAA 1","Lac","Lip"]
logging.basicConfig(filename='modele.txt', level=logging.INFO, format='%(asctime)s %(message)s')
"""if train:
    
    for epoch in range(epochs):
        max_loss_amplitude = 0.0
        max_loss_Width = 0.0
        max_loss_type=0.0
        max_loss_spectrum=0.0
        val_max_loss_amplitude = 0.0
        val_max_loss_Width = 0.0
        val_max_loss_type=0.0
        val_max_loss_spectrum=0.0
        all_predictions = []
        all_true_values = []
        true_areas = []
        pred_areas = []
        val_true_areas = []
        val_pred_areas = []
        model.train()  # Set model to training mode
    

        # Iterate over the batches in loader_train
        for batch_index, batch in enumerate(loader_train):
            input_spectre, amplitudes, widths, centers,types = batch
            input_spectre = input_spectre.unsqueeze(1)
            amplitude, width, reconstructed_spectrum,type = model(input_spectre)

            amplitudes = torch.stack(amplitudes).float().t()
            widths = torch.stack(widths).float().t()
            centers = torch.stack(centers).float().t()
            types = torch.stack(types).float().t()

            optimizer.zero_grad()
            # Calculate losses
            loss_amplitude = criterion_amplitude(amplitude, amplitudes)
            loss_Width = criterion_Width(width, widths)
            loss_type = criterion_type(type, types)
            reconstructed_spectrum_resized = F.interpolate(reconstructed_spectrum, size=input_spectre.size(2), mode='linear', align_corners=False)
            loss_spectrum = criterion_spectrum(reconstructed_spectrum_resized, input_spectre)


            max_loss_amplitude = max(max_loss_amplitude, loss_amplitude.item())
            max_loss_Width = max(max_loss_Width, loss_Width.item())
            max_loss_type = max(max_loss_type, loss_type.item())
            max_loss_spectrum = max(max_loss_spectrum, loss_spectrum.item())

            normalized_loss_amplitude = loss_amplitude / max_loss_amplitude
            normalized_loss_Width = loss_Width / max_loss_Width
            normalized_loss_type = loss_type / max_loss_type
            normalized_loss_spectrum =loss_spectrum / max_loss_spectrum
            
            
            # Combine losses with weights
            loss_boundaries =normalized_loss_spectrum*spectrum_Facteur
        # Backpropagation
            loss_boundaries.backward()
            #aire sous courbe
            for i in range(input_spectre.shape[0]):
                for j in range(amplitudes.shape[1]):
                    if types[i,j] == 1:  # Assuming '1' represents Gaussian type
                        true_area = gaussian_area(amplitudes[i, j].item(), widths[i, j].item())
                        pred_area = gaussian_area(amplitude[i, j].item(), width[i, j].item())
                    else:  # Lorentzian
                        true_area = lorentzian_area(amplitudes[i, j].item())
                        pred_area = lorentzian_area(amplitude[i, j].item())

                    true_areas.append(true_area)
                    pred_areas.append(pred_area)

            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()

    
        true_areas_np = np.array(true_areas)
        pred_areas_np = np.array(pred_areas)
        if true_areas_np.ndim == 1:
            true_areas_np = true_areas_np.reshape(1, -1)
        if pred_areas_np.ndim == 1:
            pred_areas_np = pred_areas_np.reshape(1, -1)
        
        pred_areas_np_reshaped = pred_areas_np.reshape(taille_train, 13)
        true_areas_np_reshaped = true_areas_np.reshape(taille_train, 13)
    
        pred_areas_np_transposed = np.transpose(pred_areas_np_reshaped)
        true_areas_np_transposed=np.transpose(true_areas_np_reshaped)
        correlation_matrix = np.corrcoef(pred_areas_np_transposed)
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
        axes = axes.flatten()

        for idx, metabolite in enumerate(metabolites):
            if true_areas_np.ndim > 1 and pred_areas_np.ndim > 1:
                if len(true_areas_np_transposed[idx]) > 0 and len(pred_areas_np_transposed[idx]) > 0:
                    axes[idx].scatter(true_areas_np_transposed[idx], pred_areas_np_transposed[idx], alpha=0.5)
                    axes[idx].plot([true_areas_np_transposed[idx].min(), true_areas_np_transposed[idx].max()], 
                                [true_areas_np_transposed[idx].min(), true_areas_np_transposed[idx].max()], 
                                'r--', lw=2)
                    axes[idx].set_xlabel('True Areas')
                    axes[idx].set_ylabel('Predicted Areas')
                    axes[idx].set_title(f'{metabolite} - Epoch {epoch}')
        for i in range(len(metabolites), len(axes)):
            fig.delaxes(axes[i])
        
        os.makedirs('plots2_ablationAWT', exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'plots2_AblationAWT/true_vs_predicted_epoch_{epoch}.png')
        plt.close()

        
        for i in range(13):
            r2_scores_all.append(r_squared(true_areas_np_transposed[i],pred_areas_np_transposed[i]))
            mse_all.append(mse(true_areas_np_transposed[i],pred_areas_np_transposed[i]))

       # Plot the correlation matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Correlation Matrix between Metabolites based on Predicted Areas (Epoch {epoch})')
        plt.xticks(np.arange(len(metabolites)), metabolites, rotation=90)
        plt.yticks(np.arange(len(metabolites)), metabolites)
        os.makedirs('corr_ablation_AWT', exist_ok=True)
        plt.savefig(f'corr_ablation_AWT/predicted_correlation_epoch_{epoch}.png')
        plt.close()

        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            
            for batch in loader_val:
                input_spectre, amplitudes, widths, centers,types = batch
                # Ensure proper conversion of list elements to tensors
                amplitudes = torch.stack([torch.tensor(a) for a in amplitudes]).t()
                types = torch.stack([torch.tensor(a) for a in types]).t()
                widths = torch.stack([torch.tensor(w) for w in widths]).t()
                centers = torch.stack([torch.tensor(c) for c in centers]).t()
                # Debugging statements
                #print("Amplitudes Shape:", amplitudes.shape)
                #print("Widths Shape:", widths.shape)
                #print("Centers Shape:", centers.shape)
                #print("Types Shape:", types.shape)
            
                input_spectre = input_spectre.unsqueeze(1)
                amplitude, width, reconstructed_spectrum ,type= model(input_spectre)
                

                # Calculate validation losses
                val_loss_amplitude = criterion_amplitude(amplitude, amplitudes)
                val_loss_Width = criterion_Width(width,widths)
                val_loss_type = criterion_type(type,types)
                reconstructed_spectrum_resized = F.interpolate(reconstructed_spectrum, size=input_spectre.size(2), mode='linear', align_corners=False)
                val_loss_spectrum = criterion_spectrum(reconstructed_spectrum_resized, input_spectre)
                
                val_max_loss_amplitude = max(val_max_loss_amplitude, val_loss_amplitude.item())
                val_max_loss_Width = max(val_max_loss_Width, val_loss_Width.item())
                val_max_loss_type = max(val_max_loss_type, val_loss_type.item())
                val_max_loss_spectrum = max(val_max_loss_spectrum, val_loss_spectrum.item())

                val_normalized_loss_amplitude = val_loss_amplitude / val_max_loss_amplitude
                val_normalized_loss_width = val_loss_Width / val_max_loss_Width
                val_normalized_loss_type = val_loss_type / val_max_loss_type
                val_normalized_loss_spectrum = val_loss_spectrum / val_max_loss_spectrum

                val_loss_boundaries = val_normalized_loss_spectrum*spectrum_Facteur
                
           
            print(f"epoch{epoch + 1} - "
            f"Loss Amplitude: {normalized_loss_amplitude.item():.4f}, "
            f"Loss Width: {normalized_loss_Width.item():.4f}, "
            f"Loss Type: {normalized_loss_type.item():.4f},"
            f"loss spectrum:{normalized_loss_spectrum.item():.4f} "
            f"loss total: {loss_boundaries}"
           
         )

   
    mean_r2_np=np.array(r2_scores_all)
    mean_r2_reshaped= mean_r2_np.reshape(epochs,13)
    mean_r2_transposed=np.transpose(mean_r2_reshaped)
    print("training r2 score")
    print(mean_r2_reshaped[-1])


    mean_mse_np=np.array(mse_all)
    mean_mse_reshaped= mean_mse_np.reshape(epochs,13)
    mean_mse_transposed=np.transpose(mean_mse_reshaped)
    print("training mse score")
    print(mean_mse_reshaped[-1])


    # Plotting the R-squared value evolution over epochs for each parameter
    fig, axes = plt.subplots(3, 5, figsize=(20, 15))  # Create 4x5 subplots
    fig.suptitle('R-squared Values Over Epochs for Each Parameter')

    axes_flat = axes.flatten()

    for metabolite_idx, metabolite in enumerate(metabolites):
        axes_flat[metabolite_idx].plot(range(1, epochs + 1), mean_r2_transposed[metabolite_idx])
        axes_flat[metabolite_idx].set_title(metabolite)
        axes_flat[metabolite_idx].set_xlabel('Epochs', fontsize=12, fontweight='bold') 
        axes_flat[metabolite_idx].set_ylabel('R2', fontsize=12, fontweight='bold')
    
        # Hide unused subplots
    for i in range(len(metabolites), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    # Adjust layout for better visualization
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 

    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.4, hspace=0.6)  
    plt.show()
    # Save the final model
    torch.save(model.state_dict(), f'')""" #path to save weights