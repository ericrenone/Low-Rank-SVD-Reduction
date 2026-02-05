#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import svd

# --- Production Ready Styling ---
try:
    plt.style.use('ggplot')
except:
    plt.style.use('bmh')

class SVDZeroLab:
    def __init__(self, size: int = 1000):
        self.size = size
        self.rng = np.random.default_rng(2026)
        
        # 1. High-Res Signal Generation
        x = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, x)
        self.L_clean = (np.sin(X*2) * np.exp(-X**2/4) * np.cos(Y*2) * np.exp(-Y**2/4))
        
        # 2. Additive White Gaussian Noise
        self.sigma = 0.1
        self.A_noisy = self.L_clean + self.sigma * self.rng.standard_normal((size, size))
        
        # 3. Spectral Decomposition
        print(f"Decomposing {size}x{size} matrix...")
        self.U, self.s, self.Vh = svd(self.A_noisy, full_matrices=False)

        # 4. Interface Setup
        self.fig = plt.figure(figsize=(16, 9))
        self.ax_3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_2d = self.fig.add_subplot(1, 2, 2)
        
        # Slider: 0 = Full Signal, 1000 = Total Zero
        ax_slide = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slide, 'Components to REMOVE', 0, size, valinit=0, valfmt='%d')
        self.slider.on_changed(self.update)
        
        self.update(0)

    def update(self, val):
        remove_count = int(self.slider.val)
        
        # Logic: Zero out the top 'remove_count' singular values
        s_modified = self.s.copy()
        if remove_count > 0:
            s_modified[:remove_count] = 0
            
        # Reconstruction from the remaining (weaker) components
        A_k = (self.U * s_modified) @ self.Vh
        
        # Plot 1: 3D Visualization (Strided for speed)
        self.ax_3d.clear()
        x_g, y_g = np.meshgrid(np.arange(0, self.size, 10), np.arange(0, self.size, 10))
        # As you remove components, the surface will flatten towards Y=0
        self.ax_3d.plot_surface(x_g, y_g, A_k[::10, ::10], cmap='viridis', antialiased=False)
        self.ax_3d.set_zlim(-1, 1)
        self.ax_3d.set_title(f"Surface with {remove_count} Components Zeroed")
        self.ax_3d.axis('off')
        
        # Plot 2: Spectral Residual
        self.ax_2d.clear()
        self.ax_2d.semilogy(self.s, color='gray', alpha=0.2, label="Original Spectrum")
        if remove_count < self.size:
            self.ax_2d.semilogy(range(remove_count, self.size), self.s[remove_count:], 
                                'b-', label="Active Components")
        
        self.ax_2d.set_title("Remaining Spectral Energy")
        self.ax_2d.set_ylabel("Magnitude (Log)")
        self.ax_2d.legend()
        
        # Terminal Feedback
        current_norm = np.linalg.norm(A_k)
        print(f"Removed: {remove_count:4d} | Residual Matrix Norm: {current_norm:.4f}")
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    lab = SVDZeroLab()
    plt.show()
