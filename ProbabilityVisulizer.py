import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors
from matplotlib import cm
# Make sure this import happens - it's required for 3D plots even if not directly used
from mpl_toolkits.mplot3d import Axes3D

def get_distribution_choice():
    print("\nAvailable distributions:")
    print("1. Normal (Gaussian)")
    print("2. Exponential")
    print("3. Gamma")
    print("4. Beta")
    print("5. Uniform")
    
    choices = []
    parameters = []
    
    for i in range(2):
        valid_choice = False
        while not valid_choice:
            try:
                print(f"\nSelect distribution {i+1} (1-5):")
                choice = int(input("> "))
                if choice < 1 or choice > 5:
                    print("Please enter a number between 1 and 5")
                    continue
                
                if choice == 1:  # Normal
                    print("Enter mean:")
                    mean = float(input("> "))
                    print("Enter standard deviation (> 0):")
                    std = float(input("> "))
                    if std <= 0:
                        print("Standard deviation must be positive")
                        continue
                    parameters.append((mean, std))
                    
                elif choice == 2:  # Exponential
                    print("Enter scale parameter (lambda) (> 0):")
                    scale = float(input("> "))
                    if scale <= 0:
                        print("Scale parameter must be positive")
                        continue
                    parameters.append((scale,))
                    
                elif choice == 3:  # Gamma
                    print("Enter shape parameter (alpha) (> 0):")
                    alpha = float(input("> "))
                    print("Enter scale parameter (> 0):")
                    scale = float(input("> "))
                    if alpha <= 0 or scale <= 0:
                        print("Parameters must be positive")
                        continue
                    parameters.append((alpha, scale))
                    
                elif choice == 4:  # Beta
                    print("Enter alpha parameter (> 0):")
                    alpha = float(input("> "))
                    print("Enter beta parameter (> 0):")
                    beta = float(input("> "))
                    if alpha <= 0 or beta <= 0:
                        print("Parameters must be positive")
                        continue
                    parameters.append((alpha, beta))
                    
                elif choice == 5:  # Uniform
                    print("Enter lower bound:")
                    a = float(input("> "))
                    print("Enter upper bound (> lower bound):")
                    b = float(input("> "))
                    if b <= a:
                        print("Upper bound must be greater than lower bound")
                        continue
                    parameters.append((a, b))
                
                choices.append(choice)
                valid_choice = True
                
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    return choices, parameters

def get_pdf(choice, params, x):
    if choice == 1:  # Normal
        mean, std = params
        return stats.norm.pdf(x, loc=mean, scale=std)
    elif choice == 2:  # Exponential
        scale = params[0]
        return stats.expon.pdf(x, scale=scale)
    elif choice == 3:  # Gamma
        a, scale = params
        return stats.gamma.pdf(x, a=a, scale=scale)
    elif choice == 4:  # Beta
        a, b = params
        # Beta is defined on [0,1], so we need to scale x to this range
        x_scaled = (x - x.min()) / (x.max() - x.min())
        return stats.beta.pdf(x_scaled, a=a, b=b)
    elif choice == 5:  # Uniform
        a, b = params
        return stats.uniform.pdf(x, loc=a, scale=b-a)

def visualize_joint_distribution(choices, parameters):
    # Generate x and y values
    dist1_name = get_dist_name(choices[0])
    dist2_name = get_dist_name(choices[1])
    
    # Determine appropriate range for each distribution
    x_min, x_max = determine_range(choices[0], parameters[0])
    y_min, y_max = determine_range(choices[1], parameters[1])
    
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate PDF values for each distribution
    pdf_x = np.array(get_pdf(choices[0], parameters[0], x))
    pdf_y = np.array(get_pdf(choices[1], parameters[1], y))
    
    # Calculate joint PDF assuming independence (product of PDFs)
    Z = np.outer(pdf_y, pdf_x)
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Create a 3D axes - FIXED: proper way to create 3D axes
    ax = fig.add_subplot(111, projection='3d')
    
    # Create rainbow colormap
    rainbow_cmap = create_colormap()
    
    # Normalize the values for coloring
    norm = mcolors.Normalize(vmin=np.min(pdf_x), vmax=np.max(pdf_x))
    
    # Create a color map for each x value
    colors = np.zeros((len(pdf_y), len(pdf_x), 4))  # RGBA values
    for i in range(len(pdf_x)):
        color = rainbow_cmap(norm(pdf_x[i]))
        for j in range(len(pdf_y)):
            colors[j, i] = color
    
    # Plot the surface with colors
    ax.plot_surface(
        X, Y, Z,
        facecolors=colors,
        alpha=0.8
    )
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=rainbow_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{dist1_name} Density')
    
    # Set labels and title
    ax.set_xlabel(f'X: {dist1_name} Distribution')
    ax.set_ylabel(f'Y: {dist2_name} Distribution')
    ax.set_zlabel('Joint Probability Density')
    ax.set_title(f'Joint Distribution of {dist1_name} and {dist2_name}')
    
    plt.tight_layout()
    plt.show()

def create_colormap():
    """Create a rainbow colormap for the visualization.
    Uses the full spectrum of colors from purple/blue (low values) to red (high values)."""
    return plt.cm.rainbow  # Using the built-in rainbow colormap

def get_dist_name(choice):
    names = {
        1: "Normal",
        2: "Exponential",
        3: "Gamma",
        4: "Beta",
        5: "Uniform"
    }
    return names.get(choice, "Unknown")

def determine_range(choice, params):
    """Determine appropriate range for a distribution based on choice and parameters."""
    if choice == 1:  # Normal
        mean, std = params
        # Use 4 standard deviations for normal distribution (covers ~99.99% of distribution)
        return mean - 4*std, mean + 4*std
    
    elif choice == 2:  # Exponential
        scale = params[0]
        # Exponential distribution has support [0, inf), use 5*scale to cover most of density
        return 0, 5*scale
    
    elif choice == 3:  # Gamma
        shape, scale = params
        # For gamma, more appropriate ranges based on shape parameter
        if shape < 1:
            return 0, 10*scale  # For small shape, density concentrated near 0 but with long tail
        else:
            # Mean of gamma is shape*scale
            mean = shape * scale
            # Variance is shape*scale^2
            std = np.sqrt(shape) * scale
            return max(0, mean - 3*std), mean + 5*std
    
    elif choice == 4:  # Beta
        # Beta is already defined on [0,1]
        return 0, 1
    
    elif choice == 5:  # Uniform
        a, b = params
        # Uniform distribution has exact support [a,b]
        return a, b
    
    # Default range if we reach here
    return -5, 5

def main():
    print("Joint Probability Distribution Visualizer")
    print("----------------------------------------")
    print("This program creates a 3D visualization of the joint probability distribution")
    print("of two continuous distributions with user-defined parameters.")
    print("The visualization will show a colored x-axis with a rainbow gradient,")
    print("with colors ranging from purple/blue (low density) to red (high density).")
    
    choices, parameters = get_distribution_choice()
    visualize_joint_distribution(choices, parameters)

if __name__ == "__main__":
    main()