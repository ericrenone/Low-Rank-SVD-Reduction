#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import svd

# Modern UI Styling
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

class SVDSotaLab:
    def __init__(self, size: int = 64):
        self.size = size
        self.rng = np.random.default_rng(2026)
        
        # 1. Signal Generation (Low-Rank Ground Truth)
        x = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, x)
        self.L_clean = (np.exp(-(X**2 + Y**2)) + 
                        0.5 * np.exp(-((X-1)**2 + (Y-1)**2)))
        
        # 2. Additive White Gaussian Noise (AWGN)
        self.sigma = 0.12
        self.A_noisy = self.L_clean + self.sigma * self.rng.standard_normal((size, size))
        
        # 3. Spectral Decomposition
        self.U, self.s, self.Vh = svd(self.A_noisy, full_matrices=False)
        self.energy = np.cumsum(self.s**2) / np.sum(self.s**2)

        # 4. Interface Initialization
        self.fig = plt.figure(figsize=(14, 7), constrained_layout=True)
        self.ax_3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_2d = self.fig.add_subplot(1, 2, 2)
        
        ax_slide = plt.axes([0.25, 0.05, 0.5, 0.03])
        self.slider = Slider(ax_slide, 'Target Rank', 1, 32, valinit=2, valfmt='%d')
        self.slider.on_changed(self.update)
        
        self.update(2)

    def update(self, val):
        k = int(self.slider.val)
        A_k = (self.U[:, :k] * self.s[:k]) @ self.Vh[:k, :]
        
        # 3D Topological Recovery
        self.ax_3d.clear()
        x_g, y_g = np.meshgrid(np.arange(self.size), np.arange(self.size))
        self.ax_3d.plot_surface(x_g, y_g, A_k, cmap='magma', antialiased=True)
        self.ax_3d.set_zlim(-0.2, 1.2)
        self.ax_3d.set_title(rf"$\text{{Rank }} {k} \text{{ Approximation}}$", fontsize=14)
        self.ax_3d.axis('off')
        
        # Spectral Decay Plot
        self.ax_2d.clear()
        self.ax_2d.semilogy(self.s, 'o-', color='lightgray', ms=3, label="Total Spectrum")
        self.ax_2d.semilogy(range(k), self.s[:k], 'ro', ms=5, label="Retained Signal")
        self.ax_2d.set_title("Singular Value Magnitude", fontsize=12)
        self.ax_2d.set_ylabel(r"$\sigma_i$ (Log Scale)")
        self.ax_2d.legend()
        
        # Aligned Terminal Print
        mse = np.mean((self.L_clean - A_k)**2)
        print(f"Rank: {k:2d} | Information: {self.energy[k-1]*100:6.2f}% | MSE: {mse:.6f}")
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print("-" * 50 + "\n SOTA AI/ML SVD DEMO ACTIVATED \n" + "-" * 50)
    lab = SVDSotaLab()
    plt.show()