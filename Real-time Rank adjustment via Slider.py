#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Interactive SVD Topographic 
──────────────────────────────────────────────
Features:
- Real-time Rank adjustment via Slider
- Gavish-Donoho Optimal Rank calculation
- Proportional 3D axes alignment
- Raw-string docstrings for Python 3.14 compatibility
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import svd

# Set visual style
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})

def create_signal(size: int = 60) -> np.ndarray:
    """Creates a complex low-rank surface with 3 distinct peaks."""
    x = np.linspace(-2.5, 2.5, size)
    y = np.linspace(-2.5, 2.5, size)
    X, Y = np.meshgrid(x, y)
    # Rank-3 signal
    Z = (np.exp(-(X**2 + Y**2)) + 
         0.7 * np.exp(-((X-1.5)**2 + (Y-1.5)**2)) + 
         0.5 * np.exp(-((X+1)**2 + (Y-1)**2)))
    return Z

def get_optimal_rank(s: np.ndarray, m: int, n: int, sigma: float) -> int:
    """Calculates the Gavish-Donoho rank."""
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = omega * sigma * np.sqrt(max(m, n))
    return int(np.sum(s > tau))

class SVDEngine:
    def __init__(self, size=60, noise_std=0.15):
        self.L_clean = create_signal(size)
        self.noise = noise_std * np.random.default_rng(2026).standard_normal(self.L_clean.shape)
        self.A_noisy = self.L_clean + self.noise
        
        # Precompute SVD
        self.U, self.s, self.Vh = svd(self.A_noisy, full_matrices=False)
        self.opt_rank = get_optimal_rank(self.s, size, size, noise_std)
        
        # Grid for plotting
        x = np.arange(size)
        self.X, self.Y = np.meshgrid(x, x)
        
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.canvas.manager.set_window_title('PhD SVD Real-Time Analysis')
        
        # Create Subplots
        self.ax_orig = self.fig.add_subplot(1, 3, 1, projection='3d')
        self.ax_noise = self.fig.add_subplot(1, 3, 2, projection='3d')
        self.ax_denoise = self.fig.add_subplot(1, 3, 3, projection='3d')
        
        # Slider setup
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='#f0f0f0')
        self.slider = Slider(ax_slider, 'Truncation Rank', 1, 30, valinit=self.opt_rank, valfmt='%d')
        self.slider.on_changed(self.update)
        
        self.draw_static()
        self.update(self.opt_rank)

    def draw_static(self):
        """Draw plots that don't change with the slider."""
        for ax, data, title in zip([self.ax_orig, self.ax_noise], 
                                  [self.L_clean, self.A_noisy], 
                                  ["True Signal (Rank 3)", "Noisy Observation"]):
            ax.plot_surface(self.X, self.Y, data, cmap='viridis', lw=0, antialiased=True)
            ax.set_title(title, weight='bold')
            ax.set_zlim(-0.2, 1.2)
            ax.axis('off')

    def update(self, val):
        k = int(self.slider.val)
        # Low-rank reconstruction
        A_k = (self.U[:, :k] * self.s[:k]) @ self.Vh[:k, :]
        
        self.ax_denoise.clear()
        self.ax_denoise.plot_surface(self.X, self.Y, A_k, cmap='magma', lw=0, antialiased=True)
        self.ax_denoise.set_title(f"SVD Denoised (Rank {k})", color='darkred', weight='bold')
        self.ax_denoise.set_zlim(-0.2, 1.2)
        self.ax_denoise.axis('off')
        
        # Print Alignment for Console
        error = np.linalg.norm(self.L_clean - A_k, ord='fro')
        print(f"Rank: {k:2d} | Frobenius Error: {error:.6f} | Gain: {100*(1-error/np.linalg.norm(self.noise)):>6.2f}%")
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print(f"{'SVD DYNAMIC RECOVERY SESSION':^50}")
    print("-" * 50)
    engine = SVDEngine()
    plt.show()
