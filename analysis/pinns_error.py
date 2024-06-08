import matplotlib.pyplot as plt
from models.pinnet import pinns_mse_to_numerical_solution

def plot_mse_hidden_dim(layer, fichier, hidden_dims_range, epochs, save_path='mse_plot.png'):
    """
    Trains multiple PINN models with varying hidden dimensions and plots the resulting maximum MSEs and training losses.
    Saves the plot to a specified file.

    Args:
    layer (int): Number of hidden layers in each model.
    file (str): Path to the CSV file containing the numerical solution data.
    hidden_dims_range (range): A range of hidden dimension values to iterate over.
    epochs (int): Number of training epochs.
    save_path (str): File path where the plot will be saved.
    """
    mse_results = []
    hidden_dims = []
    loss_results = []

    for hid_dim in hidden_dims_range:
        print(f'Training with {hid_dim} hidden neurons:')
        mse_max, loss = pinns_mse_to_numerical_solution(hid_dim, layer, epochs, fichier)
        print(f'Maximum MSE for {hid_dim} hidden neurons: {mse_max}')
        print(f'Training loss for {hid_dim} hidden neurons: {loss}')
        mse_results.append(mse_max)
        hidden_dims.append(hid_dim)
        loss_results.append(loss)

    # Plotting both MSE results and loss results
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Hidden Neurons')
    ax1.set_ylabel('Maximum MSE', color=color)
    ax1.plot(hidden_dims, mse_results, marker='o', linestyle='-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Training Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(hidden_dims, loss_results, marker='x', linestyle='--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Maximum MSE and Training Loss vs. Number of Hidden Neurons')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f'Plot saved to {save_path}')

# Usage example:
epochs = 400
layers = 3  # Number of hidden layers
file = "simulation_data.csv"
hidden_dims_range = range(10, 100, 2)
plot_mse_hidden_dim(layers, file, hidden_dims_range,epochs)