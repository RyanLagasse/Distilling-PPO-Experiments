import numpy as np
import matplotlib.pyplot as plt

# Distilled model sizes and their mean/std rewards
sizes = np.array([64, 32, 16, 8, 4, 2, 1])
mean_rewards = np.array([-113.438009, -115.259620,
                         151.312019, 180.801025, 162.298372, 105.768996, -657.667694])
std_rewards = np.array([81.304396, 100.337313,
                        106.424486, 50.467088, 124.501188, 137.095728, 87.477266])

# Fit a degree-2 polynomial curve
coeffs = np.polyfit(sizes, mean_rewards, deg=2)
poly = np.poly1d(coeffs)
x_smooth = np.linspace(sizes.min(), sizes.max(), 200)
y_smooth = poly(x_smooth)

# Plot 1: performance vs size with fitted curve
plt.figure()
plt.scatter(sizes, mean_rewards)
plt.plot(x_smooth, y_smooth)
plt.xscale('linear')
plt.xlabel('Hidden units')
plt.ylabel('Mean Reward')
plt.title('Performance vs Network Size')
plt.tight_layout()
plt.savefig('performance_vs_size.png')
plt.close()

# Plot 2: performance vs size with error bars and fitted curve
plt.figure()
plt.errorbar(sizes, mean_rewards, yerr=std_rewards, fmt='o')
plt.plot(x_smooth, y_smooth)
plt.xscale('linear')
plt.xlabel('Hidden units')
plt.ylabel('Mean Reward')
plt.title('Performance vs Network Size with Error Bounds')
plt.tight_layout()
plt.savefig('performance_vs_size_with_error_bounds.png')
plt.close()

# Done - images saved

