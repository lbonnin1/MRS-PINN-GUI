import numpy as np
import matplotlib.pyplot as plt
import pydicom as py
from scipy.fft import fft, ifft


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Traitement spectre brut 1H-SRM7T : load et TF

def load_dcm_data(path):
    table_data = []
    #Lecture tag dicom grâce a pydicom
    ds = py.dcmread(path)
    #print(ds)
    #Nombres de voxels par lignes
    rows = int(ds.Rows)
    #Nombres de voxels par colonnes
    columns = int(ds.Columns)
    
    spectro_data = ds.SpectroscopyData
    #Nombres de points par voxel (1024) (on obtient ce chiffre par calcul)
    points= int(len(np.frombuffer(spectro_data, np.float32))/rows/columns)
    # print("Lecture des données en cours :")
    table_data.append(np.frombuffer(spectro_data, np.float32))#like reshape but directly a vector
    # print('table_data',table_data)
    # print('table_data', type(table_data))
    # print('table_data',np.shape(table_data))

    return table_data, points, rows, columns

def tf_data(table_data, rows, columns, points):
    print("Transformée de fourier en cours :")
    table_data_itere = []
    table_data_complex = []
    tabledatatfreshape = []
    voxels = rows*columns
    tabledatatfreshape = np.zeros((int(points/2),rows,columns),dtype=complex)
    idz=0
    for a in table_data :
        for i in range (voxels) :
            table_data_itere=a[i*points:(i+1)*points]
            table_data_complex=[complex(table_data_itere[i], table_data_itere[i+1]) for i in range(0,len(table_data_itere),2)]
            idx=np.array(np.unravel_index(i,(rows,columns)))
            #Transformé de fourier (shift avant pour reordonner les fréquences)
            tabledatatfreshape[:,idx[0],idx[1]]=np.fft.fftshift(np.fft.fft(table_data_complex))
        idz=idz+1

    tabledatatfreshape_reverse = tabledatatfreshape[::-1] 

    spectre_inversé_absolu = np.abs(tabledatatfreshape_reverse[:, 0, 0])
    spectre_inversé = tabledatatfreshape_reverse[:, 0, 0]

    # Calculer la transformée de Fourier du spectre
    spectre_fft = fft(spectre_inversé_absolu)

    # Déterminer les fréquences associées à chaque point de la transformée de Fourier
    n = len(spectre_inversé_absolu)
    frequence = np.fft.fftfreq(n)

    # Supprimer les composantes de basse fréquence correspondant à la ligne de base (par exemple, les 10 premières fréquences)
    composantes_ligne_base = 10
    spectre_fft[:composantes_ligne_base] = 0

    # Calculer la transformée de Fourier inverse pour revenir dans le domaine initial
    spectre_corrige_methode_5 = ifft(spectre_fft).real
    return spectre_corrige_methode_5

def traitement (spectre_corrige_methode_5, valeurs_numeriques_PPM):

    # Fonction pour trouver le prochain maximum
    def find_next_max(spectrum, ppm_values):
        max_index = np.argmax(np.abs(spectrum))
        max_value = np.abs(spectrum[max_index])
        max_ppm = ppm_values[max_index]
        return max_value, max_index, max_ppm

    # Trouver et exclure les maximums successifs
    max_values = []
    max_indices = []
    max_ppms = []
    spectre_find_max = spectre_corrige_methode_5.copy()
    i = 5
    n = len(spectre_corrige_methode_5)

    for _ in range(i):
        max_value, max_index, max_ppm = find_next_max(spectre_find_max, valeurs_numeriques_PPM)
        
        if max_ppm >= 4:
            spectre_find_max[valeurs_numeriques_PPM >= 4] = 0
            i += 1  # Increase i to ensure we find the correct number of peaks
        else:
            lower_bound = max(0, max_index - 40)
            upper_bound = min(n, max_index + 40)
            spectre_find_max[lower_bound:upper_bound] = 0
            max_values.append(max_value)
            max_indices.append(max_index)
            max_ppms.append(max_ppm)

    # Vérifier et afficher les maximums
    for i, max_ppm in enumerate(max_ppms):
        if 4 <= max_ppm <= 5:
            spectre_corrige_methode_5[(valeurs_numeriques_PPM >= 4) & (valeurs_numeriques_PPM <= 5)] = 0
            print(f"Le maximum {i+1} est compris entre 4 et 5 ppm : c'est la raie de l'eau.")
        else:
            print(f"Le maximum {i+1} n'est pas compris entre 4 et 5 ppm : {max_ppm} ppm.")

    # Trouver les couples de max_ppm avec une différence comprise entre 0.17 et 0.22
    ppm_pairs = []
    for i in range(len(max_ppms)):
        for j in range(i + 1, len(max_ppms)):
            if 0.17 <= abs(max_ppms[i] - max_ppms[j]) <= 0.22:
                ppm_pairs.append((max_ppms[i], max_ppms[j]))

    print("Les couples de max_ppm avec une différence comprise entre 0.17 et 0.22 sont :")
    for pair in ppm_pairs:
        print(pair)

    # Trouver le plus grand PPM du pair avec la différence comprise entre 0.17 et 0.22
    if ppm_pairs:
        ppm_couple = max(ppm_pairs[0])  # Utiliser le plus grand PPM du premier couple trouvé
    else:
        raise Exception("No valid pair found within the specified difference range.")

    # Calculer le décalage nécessaire pour déplacer le maximum à 3.21 ppm
    shift_amount = int((3.21 - ppm_couple) / (valeurs_numeriques_PPM[1] - valeurs_numeriques_PPM[0]))

    # Décaler les données du spectre pour que le maximum soit à 3.21 ppm
    spectre_corrige_methode_5_shifted = np.roll(spectre_corrige_methode_5, shift_amount)
    spectre_corrige_methode_5_shifted = np.abs(spectre_corrige_methode_5_shifted)

    return spectre_corrige_methode_5_shifted