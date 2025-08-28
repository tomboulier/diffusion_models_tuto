"""
Utilitaires pour les modèles de diffusion.

Ce module contient des fonctions helper pour travailler avec les modèles de diffusion.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from PIL import Image


def create_noise_schedule(num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Crée un planning de bruit linéaire pour le processus de diffusion.
    
    Args:
        num_steps: Nombre d'étapes de diffusion
        beta_start: Valeur de départ de beta
        beta_end: Valeur finale de beta
        
    Returns:
        Tensor contenant les valeurs de beta
    """
    return torch.linspace(beta_start, beta_end, num_steps)


def add_noise(x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Ajoute du bruit à une image selon l'étape de diffusion t.
    
    Args:
        x_0: Image originale
        t: Étape de diffusion
        noise: Bruit à ajouter (généré automatiquement si None)
        
    Returns:
        Image bruitée
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    
    # Calcul des coefficients alpha
    betas = create_noise_schedule()
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t])
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod[t])
    
    return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise


def visualize_diffusion_process(image: np.ndarray, steps: List[int] = [0, 100, 300, 500, 800, 999]) -> None:
    """
    Visualise le processus de diffusion à différentes étapes.
    
    Args:
        image: Image d'entrée (numpy array)
        steps: Liste des étapes à visualiser
    """
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))
    
    # Conversion en tensor PyTorch
    if isinstance(image, np.ndarray):
        x_0 = torch.from_numpy(image).float()
    else:
        x_0 = image
    
    for i, step in enumerate(steps):
        t = torch.tensor([step])
        noisy_image = add_noise(x_0, t)
        
        # Conversion pour affichage
        display_image = noisy_image.numpy()
        if len(display_image.shape) == 3:
            display_image = np.transpose(display_image, (1, 2, 0))
        
        axes[i].imshow(display_image, cmap='gray' if len(display_image.shape) == 2 else None)
        axes[i].set_title(f'Étape {step}')
        axes[i].axis('off')
    
    plt.suptitle('Processus de diffusion directe')
    plt.tight_layout()
    plt.show()


def save_generated_images(images: List[Image.Image], prompts: List[str], save_dir: str = "generated_images") -> None:
    """
    Sauvegarde une liste d'images générées avec leurs prompts.
    
    Args:
        images: Liste d'images PIL
        prompts: Liste des prompts correspondants
        save_dir: Répertoire de sauvegarde
    """
    import os
    
    # Création du répertoire s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        # Nettoyage du prompt pour le nom de fichier
        clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '_')).rstrip()
        clean_prompt = clean_prompt.replace(' ', '_')[:50]  # Limitation de la longueur
        
        filename = f"{i+1:02d}_{clean_prompt}.png"
        filepath = os.path.join(save_dir, filename)
        
        image.save(filepath)
        print(f"Image sauvegardée: {filepath}")


def compare_models_output(prompts: List[str], models: dict) -> None:
    """
    Compare les sorties de différents modèles sur les mêmes prompts.
    
    Args:
        prompts: Liste de prompts à tester
        models: Dictionnaire {nom_modele: pipeline}
    """
    num_models = len(models)
    num_prompts = len(prompts)
    
    fig, axes = plt.subplots(num_prompts, num_models, figsize=(4*num_models, 4*num_prompts))
    
    if num_prompts == 1:
        axes = axes.reshape(1, -1)
    if num_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, prompt in enumerate(prompts):
        for j, (model_name, pipeline) in enumerate(models.items()):
            try:
                image = pipeline(prompt, num_inference_steps=20).images[0]
                axes[i, j].imshow(image)
                axes[i, j].set_title(f"{model_name}\n'{prompt}'", fontsize=10)
                axes[i, j].axis('off')
            except Exception as e:
                axes[i, j].text(0.5, 0.5, f"Erreur:\n{str(e)[:50]}...", 
                               ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f"{model_name} - Erreur")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


class DiffusionTrainer:
    """
    Classe helper pour l'entraînement de modèles de diffusion simples.
    """
    
    def __init__(self, num_steps: int = 1000):
        self.num_steps = num_steps
        self.betas = create_noise_schedule(num_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffusion(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processus de diffusion directe.
        
        Returns:
            Tuple (image_bruitée, bruit_ajouté)
        """
        noise = torch.randn_like(x_0)
        x_t = add_noise(x_0, t, noise)
        return x_t, noise
    
    def get_loss(self, model, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calcule la perte pour l'entraînement.
        """
        x_t, noise = self.forward_diffusion(x_0, t)
        predicted_noise = model(x_t, t)
        return torch.nn.functional.mse_loss(predicted_noise, noise)


if __name__ == "__main__":
    # Test des fonctions
    print("Test des utilitaires de diffusion...")
    
    # Test de création de planning de bruit
    betas = create_noise_schedule()
    print(f"Planning de bruit créé: {betas[:5]}...{betas[-5:]}")
    
    # Test d'ajout de bruit
    test_image = torch.randn(3, 64, 64)
    noisy_image = add_noise(test_image, torch.tensor([500]))
    print(f"Image bruitée créée, shape: {noisy_image.shape}")
    
    print("Tests terminés avec succès !")
