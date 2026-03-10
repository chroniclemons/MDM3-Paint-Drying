import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


evaporation_rate = 0.02
absorbency_base = 0.4
flow_rate_weight = 0.25
resivoir_capacity = 200.0
resivoir_refill_rate = 0.2

class FiberInkSim:
    def __init__(self, size=100):
        self.size = size
        # Eq. (1): Random absorbency (Base + Var) [cite: 81]
        self.absorbency = absorbency_base + 0.3 * (np.random.rand(size, size) - 0.5)
        
        # Section 3.3: Paper Texture Mask (Alignment of fibers) 
        # We generate a random vector field to simulate "fiber directions"
        self.fiber_weights = np.random.rand(size, size, 8) 
        self.water = np.zeros((size, size))
        self.carbon = np.zeros((size, size))
        self.reservoir = np.zeros((size, size))
        
    def apply_drop(self, cx, cy, radius=3):
        y, x = np.ogrid[:self.size, :self.size]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.reservoir[mask] = resivoir_capacity # High water volume for gradient [cite: 198]
        self.carbon[mask] = 255.0

    def update(self):
        # Section 4.3: Continuous Ink Refilling [cite: 245]
        refill = self.reservoir * resivoir_refill_rate
        self.water += refill
        self.reservoir -= refill
        
        next_w = self.water.copy()
        next_c = self.carbon.copy()
        
        # Iterate through active papels [cite: 75]
        rows, cols = np.where(self.water > 0.1)
        for r, c in zip(rows, cols):
            if 0 < r < self.size-1 and 0 < c < self.size-1:
                # Eq. (2): Flow rate K(p) [cite: 95]
                k_p = flow_rate_weight * self.absorbency[r, c]
                total_out = self.water[r, c] * k_p
                
                neighbors = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
                scores = []
                for i, (dr, dc) in enumerate(neighbors):
                    nr, nc = r+dr, c+dc
                    # Eq. (4): Gradient (Brownian motion) [cite: 153]
                    grad = max(0, self.water[r, c] - self.water[nr, nc])
                    # Eq. (6): Absorbency [cite: 170]
                    abs_val = self.absorbency[nr, nc]
                    # Section 3.3: Paper Texture alignment weight 
                    tex = self.fiber_weights[r, c, i]
                    
                    # Combined Probability R_k [cite: 188]
                    scores.append(0.5 * grad + 0.2 * abs_val + 0.3 * tex)
                
                sum_s = sum(scores)
                if sum_s > 0:
                    for i, (dr, dc) in enumerate(neighbors):
                        nr, nc = r+dr, c+dc
                        prob = scores[i] / sum_s
                        flow = total_out * prob
                        
                        next_w[r, c] -= flow
                        next_w[nr, nc] += flow
                        
                        # Section 2.3: Carbon move with filtering effect [cite: 108]
                        if self.absorbency[nr, nc] > 0.25:
                            c_flow = flow * (self.carbon[r, c] / (self.water[r, c] + 1e-6))
                            next_c[r, c] -= c_flow
                            next_c[nr, nc] += c_flow
                            
        self.water = next_w * (1-evaporation_rate) # Evaporation [cite: 240]
        self.carbon = next_c

# Animation
def main():
    sim = FiberInkSim(size=300)
    sim.apply_drop(150, 150, radius=30)

    
    fig, ax = plt.subplots(figsize=(15,15))
    img = ax.imshow(255 - np.clip(sim.carbon * 3, 0, 255), cmap='gray', interpolation='bicubic')
    step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='red', fontweight='bold')
    def animate(i):
        sim.update()
        img.set_array(255 - np.clip(sim.carbon * 3, 0, 255))
        step_text.set_text(f"Step: {i}")
        return [img, step_text]
    ani = FuncAnimation(fig, animate, frames=200, interval=10, blit=True)
    
    
    plt.show()

def show_absorbency(sim):
    plt.imshow(sim.absorbency, cmap='viridis')
    plt.colorbar(label='Absorbency')
    plt.title('Random Absorbency Map of Paper')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()