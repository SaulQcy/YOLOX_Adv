import torch
import torch.nn.functional as F
import random

def random_pixel_perturbation(img_tensor, epslon=5, rate=0.5):
    # Create a copy of the input tensor
    t_min = torch.min(img_tensor)
    t_max = torch.max(img_tensor)
    # print(t_max)
    perturbed_img = img_tensor.clone()
    b, c, h, w = perturbed_img.shape
    
    # Generate random mask for pixels to perturb (50% probability)
    mask = torch.rand_like(perturbed_img) < rate
    
    # Generate random perturbations in [-epslon/255, epslon/255]
    # We use (2 * epslon + 1) possible values (-epslon, ..., 0, ..., epslon)/255
    perturbations = (torch.randint(-epslon, epslon + 1, (b, c, h, w), dtype=torch.float32).to(img_tensor))
    if t_max <= 1:
        perturbations /= 255.
    
    # Apply perturbations only to masked pixels
    perturbed_img[mask] += perturbations[mask]
    
    # Clip to maintain valid pixel range
    perturbed_img = torch.clamp(perturbed_img, t_min, t_max)
    
    return perturbed_img.to(img_tensor)