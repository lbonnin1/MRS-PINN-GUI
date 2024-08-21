import torch
import numpy as np

from scipy.stats import entropy

def r_squared(y_true, y_pred):
    mean_observed = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_observed)**2)
    ss_res = np.sum((y_true - y_pred)**2)
   
    if ss_tot == 0:
        if np.allclose(y_true, y_pred):
            return 1.0  # Perfect fit case
        else:
            return float('nan')  # Undefined R-squared
    
    r2 = 1 - (ss_res / ss_tot)
    return r2
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true (array-like): Array of actual values.
        y_pred (array-like): Array of predicted values.
        
    Returns:
        float: MAPE value.
    """
    nonzero_mask = y_true != 0
    y_true, y_pred = y_true[nonzero_mask], y_pred[nonzero_mask]
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def gaussian_area(amplitude, lwhm):
    # Constante sqrt(pi) / sqrt(ln(2))
    sqrt_pi = np.sqrt(np.pi)
    sqrt_ln2 = np.sqrt(np.log(2))
    
    # Calcul de l'aire sous la courbe gaussienne
    area = amplitude * (lwhm * sqrt_pi) / (2 * sqrt_ln2)
    
    return area

def lorentzian_area(amplitude):
    # Pour une lorentzienne, l'aire sous la courbe est simplement l'amplitude
    return amplitude


def draw_curve(x, amplitude, center, lwhm, type, epsilon=1e-10):
    """
    Draws a curve based on the specified type (Lorentzian or Gaussian).

    Args:
        x (np.ndarray): The x-values (e.g., ppm data).
        amplitude (float): The amplitude of the curve.
        center (float): The center position of the curve.
        lwhm (float): The full width at half maximum of the curve.
        type (str): The type of the curve ("Lorentzienne" or "Gaussian").
        epsilon (float): A small value to avoid division by zero (default: 1e-10).

    Returns:
        np.ndarray: The calculated y-values for the curve.
    """
    lwhm = lwhm if lwhm != 0 else epsilon  # Ensure lwhm is not zero to avoid division by zero
    center = center.numpy() if isinstance(center, torch.Tensor) else center
    amplitude = amplitude.numpy() if isinstance(amplitude, torch.Tensor) else amplitude

    if type == "Lorentzienne":
        return amplitude / (1 + ((x - center) / (lwhm / 2)) ** 2)
    else:
        return amplitude * np.exp(-4 * np.log(2) * ((x - center) / lwhm) ** 2)

def mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    variance = np.var(y_true)
    return mse / variance


# KLPD normalisé de 0 a 1

def kl_divergence(s1, s2, epsilon=1e-10):
    """
    Compute the KL divergence between two spectra s1 and s2.
    Args:
        s1: Tensor of shape (N,), spectrum 1
        s2: Tensor of shape (N,), spectrum 2
        epsilon: Small value to avoid division by zero and log(0)
    Returns:
        KL divergence value
    """
    s1 = s1 + epsilon
    s2 = s2 + epsilon
    kl_div = torch.sum(s1 * torch.log(s1 / s2))
    return kl_div

def spectral_kl_pseudo_divergence(s1, s2, scale_factor=1e6, epsilon=1e-10):
    """
    Compute the spectral KL pseudo-divergence between two spectra s1 and s2.
    Args:
        s1: Tensor of shape (N,), spectrum 1
        s2: Tensor of shape (N,), spectrum 2
        scale_factor: Scaling factor to normalize spectra
        epsilon: Small value to avoid division by zero and log(0)
    Returns:
        Spectral KL pseudo-divergence value
    """
    # Convert input lists to tensors
    s1 = torch.tensor(s1, dtype=torch.float64)
    s2 = torch.tensor(s2, dtype=torch.float64)
    
    # Apply scaling factor
    s1 = s1 / scale_factor
    s2 = s2 / scale_factor

    k1 = torch.sum(s1)
    k2 = torch.sum(s2)

    # Normalize the spectra
    s1_norm = s1 / k1
    s2_norm = s2 / k2

    # Compute KL divergence on normalized spectra
    kl_s1_s2 = kl_divergence(s1_norm, s2_norm, epsilon)
    kl_s2_s1 = kl_divergence(s2_norm, s1_norm, epsilon)
   
    # Compute spectral KL pseudo-divergence
    divergence = k1 * kl_s1_s2 + k2 * kl_s2_s1 + (k1 - k2) * torch.log(k1 / k2 + epsilon)

    return divergence

def normalize_klpd(klpd_value, min_klpd, max_klpd):
    """
    Normalize KLPD value to a scale from 0 to 1.
    Args:
        klpd_value: The KLPD value to normalize
        min_klpd: Minimum KLPD value (reference for 0)
        max_klpd: Maximum KLPD value (reference for 1)
    Returns:
        Normalized KLPD value between 0 and 1
    """
    return (klpd_value - min_klpd) / (max_klpd - min_klpd)

def test_klpd(spectre, fit, scale_factor=1e6, epsilon=1e-10):
    """
    Test and validate the KLPD calculation and print results normalized between 0 and 1.
    Args:
        spectre: Tensor of shape (N,), spectrum 1
        fit: Tensor of shape (N,), spectrum 2
        scale_factor: Scaling factor to normalize spectra
        epsilon: Small value to avoid division by zero and log(0)
    """
    # Calcul de la KLPD pour les spectres donnés
    klpd_fit = spectral_kl_pseudo_divergence(spectre, fit, scale_factor, epsilon)
    print("KLPD for fit:", klpd_fit.item())
    
    # Test avec des spectres identiques
    s_identique = [1.0] * len(spectre)
    klpd_identique = spectral_kl_pseudo_divergence(s_identique, s_identique, scale_factor, epsilon)
    print("KLPD for identical spectra:", klpd_identique.item())
    
    # Test avec des spectres très différents
    s_diff = [1.0] * (len(spectre) // 2) + [0.0] * (len(spectre) // 2)
    klpd_diff = spectral_kl_pseudo_divergence(spectre, s_diff, scale_factor, epsilon)
    print("KLPD for very different spectra:", klpd_diff.item())
    
    # Définir les valeurs minimales et maximales pour la normalisation
    min_klpd_value = klpd_identique.item()  # La KLPD pour les spectres identiques
    max_klpd_value = klpd_diff.item()  # La KLPD pour les spectres très différents
    
    # Normaliser les valeurs de KLPD
    klpd_fit_normalized = normalize_klpd(klpd_fit, min_klpd_value, max_klpd_value)
    print(f"KLPD for fit (normalized): {klpd_fit_normalized:.2f}")
    
    klpd_identique_normalized = normalize_klpd(klpd_identique, min_klpd_value, max_klpd_value)
    print(f"KLPD for identical spectra (normalized): {klpd_identique_normalized:.2f}")

    klpd_diff_normalized = normalize_klpd(klpd_diff, min_klpd_value, max_klpd_value)
    print(f"KLPD for very different spectra (normalized): {klpd_diff_normalized:.2f}")


def calculer_residu_normalise(input_spectre, fit):
    """
    Calcule le résidu entre le spectre d'entrée et le fit, puis normalise ce résidu.

    Paramètres:
    - input_spectre : Tensor ou numpy array du spectre d'entrée.
    - fit : Tensor ou numpy array du fit.

    Retourne:
    - mean_residu_norm_max : Moyenne du résidu normalisé par la valeur maximale du résidu.
    - mean_residu_norm_total : Moyenne du résidu normalisé par la somme totale des valeurs du spectre d'entrée.
    """
    # Vérifier si input_spectre est un tenseur PyTorch et le convertir en array numpy
    if torch.is_tensor(input_spectre):
        input_spectre_np = input_spectre.squeeze().cpu().numpy()
    else:
        input_spectre_np = input_spectre

    # Vérifier si fit est un tenseur PyTorch et le convertir en array numpy
    if torch.is_tensor(fit):
        fit_np = fit.squeeze().cpu().numpy()
    else:
        fit_np = fit

    # Calculer le résidu (différence absolue entre le spectre d'entrée et le fit)
    residu = np.abs(input_spectre_np - fit_np)

    # Normaliser le résidu par la valeur maximale du résidu
    residu_norm_max = residu / np.max(residu)

    # Normaliser le résidu par la somme totale des valeurs du spectre d'entrée
    residu_norm_total = residu / np.sum(input_spectre_np)

    # Calculer la moyenne des résidus normalisés
    mean_residu_norm_max = np.mean(residu_norm_max)
    mean_residu_norm_total = np.mean(residu_norm_total)

    return mean_residu_norm_max, mean_residu_norm_total