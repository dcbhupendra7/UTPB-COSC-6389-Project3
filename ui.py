import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_classifier_nn import load_dataset, compute_accuracy, ConvLayer, FullyConnected, SoftmaxLayer  
from sklearn.model_selection import train_test_split

class TrainingVisualizer:
    def __init__(self, master):
        # Main window setup
        self.master = master
        self.master.title("Neural Network Training Visualization")
        self.master.geometry("900x700")

        # Frames
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Add matplotlib plots
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.loss_ax = self.figure.add_subplot(211)
        self.acc_ax = self.figure.add_subplot(212)

        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epochs")
        self.loss_ax.set_ylabel("Loss")

        self.acc_ax.set_title("Validation Accuracy")
        self.acc_ax.set_xlabel("Epochs")
        self.acc_ax.set_ylabel("Accuracy (%)")

        # Adjust layout to fix overlapping text
        self.figure.tight_layout(pad=3.0)
        self.figure.subplots_adjust(hspace=0.5)

        self.loss_plot = None
        self.acc_plot = None

        self.canvas = FigureCanvasTkAgg(self.figure, self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Info Frame for epoch details
        self.info_frame = ttk.Frame(self.master)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_label = ttk.Label(self.info_frame, text="Training Progress: Waiting to start...", anchor="w")
        self.info_label.pack(side=tk.LEFT, padx=5)

        # Training data storage
        self.training_loss = []
        self.validation_accuracy = []

    def update_metrics(self, epoch, loss, accuracy):
        """Update training metrics in real-time."""
        self.training_loss.append(loss)
        self.validation_accuracy.append(accuracy)

        # Update loss plot
        if self.loss_plot:
            self.loss_plot.remove()
        self.loss_plot, = self.loss_ax.plot(range(1, len(self.training_loss) + 1), self.training_loss, label="Loss", color="blue")

        # Update accuracy plot
        if self.acc_plot:
            self.acc_plot.remove()
        self.acc_plot, = self.acc_ax.plot(range(1, len(self.validation_accuracy) + 1), self.validation_accuracy, label="Accuracy", color="green")

        self.loss_ax.legend()
        self.acc_ax.legend()

        # Refresh canvas and flush events for responsiveness
        self.canvas.draw()
        self.canvas.flush_events()

        # Update progress in info label
        self.info_label.config(text=f"Training Progress: Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    def finalize(self):
        """Finalize the training process."""
        self.info_label.config(text="Training Completed!")
        self.canvas.draw()

# Training logic
def train(network, X_train, y_train, X_val, y_val, batch_size=32, epochs=10, visualizer=None):
    num_batches = len(X_train) // batch_size

    for epoch in range(epochs):
        total_loss = 0
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size

            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            output = X_batch
            for layer in network[:-1]:  # Exclude softmax layer
                output = layer.forward(output)

            predictions = network[-1].forward(output)
            loss = -np.mean(np.log(predictions[range(len(y_batch)), y_batch] + 1e-10))
            total_loss += loss

            network[-1].backward(y_batch)
            d_output = network[-1].input

            for layer in reversed(network[:-1]):
                d_output = layer.backward(d_output)

        # Validation
        val_output = X_val
        for layer in network[:-1]:
            val_output = layer.forward(val_output)

        val_predictions = network[-1].forward(val_output)
        val_accuracy = compute_accuracy(val_predictions, y_val)

        # Update the visualizer
        if visualizer:
            visualizer.update_metrics(epoch + 1, total_loss / num_batches, val_accuracy)

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / num_batches:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    if visualizer:
        visualizer.finalize()

# Main function
def main():
    # Load dataset
    dataset_path = "dataset"
    X, y = load_dataset(dataset_path, max_images=1000)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the network
    network = [
        ConvLayer(8, 3, 1),
        FullyConnected(8 * 26 * 26, 128),
        SoftmaxLayer(128, 10)
    ]

    # Setup Tkinter UI
    root = tk.Tk()
    visualizer = TrainingVisualizer(root)

    # Start training
    root.after(100, train, network, X_train, y_train, X_val, y_val, 32, 10, visualizer)
    root.mainloop()

if __name__ == "__main__":
    main()
