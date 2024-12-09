import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Convolutional Layer
class ConvLayer:
    def __init__(self, num_filters, filter_size, input_depth):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size) / np.sqrt(input_depth * filter_size * filter_size)
        self.biases = np.zeros((num_filters, 1, 1))
        self.input = None
        self.output = None

    def forward(self, input):
        if len(input.shape) != 4:
            raise ValueError(f"Expected input shape (batch_size, channels, height, width), got {input.shape}")
        
        batch_size, channels, height, width = input.shape
        f = self.filter_size
        if height - f + 1 <= 0 or width - f + 1 <= 0:
            raise ValueError(f"Input dimensions ({height}, {width}) are too small for filter size {f}.")
        
        self.input = input
        output_height = height - f + 1
        output_width = width - f + 1
        self.output = np.zeros((batch_size, self.num_filters, output_height, output_width))
        
        for b in range(batch_size):  # Batch
            for i in range(self.num_filters):  # Filters
                for j in range(output_height):  # Slide vertically
                    for k in range(output_width):  # Slide horizontally
                        self.output[b, i, j, k] = np.sum(
                            self.filters[i] * input[b, :, j:j+f, k:k+f]
                        )
                # Add biases
                self.output[b, i] += self.biases[i]
        
        return relu(self.output)

    def backward(self, d_output):
        if d_output.shape != self.output.shape:
            d_output = d_output.reshape(self.output.shape)
        
        d_output = d_output * relu_derivative(self.output)
        batch_size, num_filters, out_height, out_width = d_output.shape
        f = self.filter_size
        input_channels = self.input.shape[1]
        
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)
        
        for b in range(batch_size):
            for i in range(num_filters):
                for c in range(input_channels):
                    for j in range(out_height):
                        for k in range(out_width):
                            d_filters[i, c] += d_output[b, i, j, k] * self.input[b, c, j:j+f, k:k+f]
                            d_input[b, c, j:j+f, k:k+f] += d_output[b, i, j, k] * self.filters[i, c]
                d_biases[i] += np.sum(d_output[b, i])
        
        learning_rate = 0.01
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        
        return d_input

# Fully Connected Layer
class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.z = None

    def forward(self, input):
        self.input = input.reshape(input.shape[0], -1)
        self.z = np.dot(self.input, self.weights) + self.biases
        return relu(self.z)

    def backward(self, d_output):
        d_output = d_output * relu_derivative(self.z)
        d_input = np.dot(d_output, self.weights.T).reshape(self.input.shape)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        
        learning_rate = 0.01
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input

# Softmax Layer
class SoftmaxLayer:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) / np.sqrt(input_size)
        self.biases = np.zeros((1, num_classes))
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input.reshape(input.shape[0], -1)
        z = np.dot(self.input, self.weights) + self.biases
        self.output = softmax(z)
        return self.output

    def backward(self, labels):
        batch_size = self.input.shape[0]
        d_output = self.output.copy()
        d_output[range(batch_size), labels] -= 1
        d_output /= batch_size
        
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_input = np.dot(d_output, self.weights.T)
        
        learning_rate = 0.01
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input

# Dataset Loader
def load_dataset(dataset_path, max_images=1000):
    images = []
    labels = []
    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                if len(images) >= max_images:
                    break
                img_path = os.path.join(label_path, img_file)
                if img_file == '.DS_Store':
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Unable to load image {img_path}. Skipping.")
                    continue
                img = cv2.resize(img, (28, 28))
                images.append(img / 255.0)
                labels.append(int(label))
        if len(images) >= max_images:
            break
    
    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded dataset with shape: {images.shape}")
    return images.reshape(-1, 1, 28, 28), labels

# Training
def compute_accuracy(predictions, labels):
    predicted_classes = np.argmax(predictions, axis=1)
    return np.mean(predicted_classes == labels) * 100

def train(network, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    num_batches = len(X_train) // batch_size
    visualizer = TrainingVisualizer()  # Initialize the visualizer

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
        visualizer.update(epoch + 1, total_loss / num_batches, val_accuracy)

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / num_batches:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    visualizer.finalize()  # Finalize the plots

if __name__ == "__main__":
    dataset_path = "dataset"
    X, y = load_dataset(dataset_path, max_images=1000)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    network = [
        ConvLayer(8, 3, 1),
        FullyConnected(8 * 26 * 26, 128),
        SoftmaxLayer(128, 10)
    ]

    train(network, X_train, y_train, X_val, y_val, batch_size=32, epochs=10)
