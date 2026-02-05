#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.linalg import svd

# Set professional scientific theme
plt.style.use('bmh')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

class SVDEducator:
    def __init__(self, size: int = 60):
        self.size = size
        self.rng = np.random.default_rng(2026)
        
        # 1. Generate Signal & Noise
        self.L_clean = self._generate_signal()
        self.sigma = 0.15
        self.noise = self.sigma * self.rng.standard_normal(self.L_clean.shape)
        self.A_noisy = self.L_clean + self.noise
        
        # 2. Decompose
        self.U, self.s, self.Vh = svd(self.A_noisy, full_matrices=False)
        self.energy_cumulative = np.cumsum(self.s**2) / np.sum(self.s**2)
        
        # 3. GUI Setup
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('SVD Informational Dynamics (Verified 2026)')
        
        self.ax_3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_spec = self.fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(bottom=0.2, wspace=0.1)
        
        # 4. Interactive Slider
        ax_rank = plt.axes([0.2, 0.08, 0.6, 0.03])
        self.slider = Slider(ax_rank, 'Rank (k)', 1, 40, valinit=3, valfmt='%d')
        self.slider.on_changed(self.update)
        
        self.update(3)

    def _generate_signal(self):
        x = np.linspace(-2, 2, self.size)
        X, Y = np.meshgrid(x, x)
        return (np.exp(-(X**2 + Y**2)) + 
                0.6 * np.exp(-((X-1.2)**2 + (Y-1.2)**2)) + 
                0.4 * np.exp(-((X+1.2)**2 + (Y-1.2)**2)))

    def update(self, val):
        k = int(self.slider.val)
        A_k = (self.U[:, :k] * self.s[:k]) @ self.Vh[:k, :]
        
        # Update 3D Surface
        self.ax_3d.clear()
        x_grid = np.arange(self.size)
        X, Y = np.meshgrid(x_grid, x_grid)
        self.ax_3d.plot_surface(X, Y, A_k, cmap='magma', antialiased=True)
        self.ax_3d.set_title(f"Rank-{k} Surface Approximation", fontweight='bold')
        self.ax_3d.set_zlim(-0.2, 1.2)
        self.ax_3d.axis('off')
        
        # Update Spectral Plot (using Raw Strings for LaTeX)
        self.ax_spec.clear()
        self.ax_spec.semilogy(self.s, 'o-', color='gray', ms=4, alpha=0.3, label='Spectrum')
        self.ax_spec.semilogy(range(k), self.s[:k], 'ro', ms=6, label='Active Signal')
        
        # Fixed labeling using Raw Strings (r"")
        self.ax_spec.set_title("Singular Value Energy Decay", fontsize=12)
        self.ax_spec.set_ylabel(r"Magnitude ($\sigma_i$)")
        self.ax_spec.set_xlabel("Component Index")
        self.ax_spec.grid(True, which='both', alpha=0.2)
        self.ax_spec.legend()
        
        energy = self.energy_cumulative[k-1] * 100
        self.ax_spec.annotate(f"Energy: {energy:.1f}%", 
                             xy=(k, self.s[k-1]), xytext=(k+5, self.s[k-1]*3),
                             arrowprops=dict(arrowstyle='->', color='black'))
        
        # Terminal Feedback (PhD Alignment)
        error = np.linalg.norm(self.L_clean - A_k, ord='fro')
        print(f"Active Rank: {k:2d} | Cumulative Energy: {energy:5.1f}% | Fro-Error: {error:.5f}")
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"{'SVD PhD LEARNING LAB: NO-WARNING EDITION':^60}")
    print(f"{'='*60}")
    lab = SVDEducator()
    plt.show()