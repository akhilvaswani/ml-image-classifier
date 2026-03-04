# ML Image Classifier

In this project, I built an image classification model using TensorFlow and Keras. The model uses a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset into 10 categories like airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

I wanted to go beyond just training a basic model, so I included data augmentation, dropout regularization, learning rate scheduling, and a full evaluation pipeline with confusion matrices and per-class accuracy breakdowns.

## What's in This Repo

- `train.py` - Main training script with CNN architecture, data augmentation, and callbacks
- `predict.py` - Load a saved model and classify new images
- `evaluate.py` - Generate detailed evaluation metrics, confusion matrix, and classification report
- `utils/data_loader.py` - Data loading and preprocessing utilities
- `utils/visualization.py` - Plotting functions for training history, confusion matrices, and sample predictions
- `config.py` - Central configuration file for hyperparameters and paths
- `requirements.txt` - Python dependencies
- `.gitignore` - Ignores model files, datasets, and virtual environments

## How I Built It

### Step 1 - Setting Up the Environment

First I created a virtual environment and installed everything I needed:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The main libraries are TensorFlow for the model, NumPy for data manipulation, Matplotlib and Seaborn for visualization, and scikit-learn for evaluation metrics.

### Step 2 - Loading and Preprocessing the Data

I used the CIFAR-10 dataset which comes built into Keras. It has 60,000 32x32 color images split across 10 classes, with 50,000 for training and 10,000 for testing.

In `utils/data_loader.py`, I normalized the pixel values to a 0-1 range and one-hot encoded the labels. I also set aside 10% of the training data as a validation set so I could monitor overfitting during training.

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

### Step 3 - Building the CNN Architecture

The model architecture in `train.py` has three convolutional blocks, each with two Conv2D layers followed by batch normalization, max pooling, and dropout. Then it flattens into two dense layers before the final softmax output.

I experimented with a few different architectures before landing on this one. The key things that helped were:
- Batch normalization after each conv layer to stabilize training
- Increasing dropout rates (0.25 after conv blocks, 0.5 before the output) to prevent overfitting
- Using 3x3 kernels consistently with "same" padding to preserve spatial dimensions

```
Input (32x32x3)
  -> Conv2D(32) -> BatchNorm -> Conv2D(32) -> BatchNorm -> MaxPool -> Dropout(0.25)
  -> Conv2D(64) -> BatchNorm -> Conv2D(64) -> BatchNorm -> MaxPool -> Dropout(0.25)
  -> Conv2D(128) -> BatchNorm -> Conv2D(128) -> BatchNorm -> MaxPool -> Dropout(0.25)
  -> Flatten -> Dense(512) -> BatchNorm -> Dropout(0.5)
  -> Dense(10, softmax)
```

### Step 4 - Data Augmentation

To help the model generalize better and reduce overfitting, I used Keras' ImageDataGenerator for real-time data augmentation during training:

```python
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
```

This randomly rotates, shifts, and flips images during training so the model sees slightly different versions of each image every epoch. It made a noticeable difference - without augmentation the model was overfitting pretty badly after about 30 epochs.

### Step 5 - Training with Callbacks

I trained the model for 100 epochs with a batch size of 64. I set up a few callbacks to make training smoother:

- **ReduceLROnPlateau** - cuts the learning rate by half if validation loss stops improving for 5 epochs
- **EarlyStopping** - stops training if validation loss hasn't improved for 15 epochs
- **ModelCheckpoint** - saves the best model based on validation accuracy

```bash
python train.py
```

Training took about 20 minutes on my machine with a GPU. The model reached around 92% validation accuracy before early stopping kicked in.

### Step 6 - Evaluating the Model

The `evaluate.py` script loads the saved model and runs a full evaluation on the test set:

```bash
python evaluate.py
```

This generates:
- Overall test accuracy and loss
- A classification report with precision, recall, and F1-score for each class
- A confusion matrix heatmap saved as `results/confusion_matrix.png`
- Per-class accuracy bar chart

From the results, the model did best on trucks and ships (around 95%) and worst on cats (around 85%), which makes sense since cats and dogs can look pretty similar at 32x32 resolution.

### Step 7 - Making Predictions on New Images

The `predict.py` script lets you classify individual images:

```bash
python predict.py --image path/to/image.jpg
```

It resizes the image to 32x32, normalizes it, runs it through the model, and returns the predicted class along with confidence scores for all 10 categories. I also added a `--visualize` flag that shows the image alongside a bar chart of the prediction probabilities.

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.8% |
| Test Loss | 0.267 |
| Best Class (ship) | 95.2% |
| Worst Class (cat) | 84.7% |
| Training Time | ~20 min (GPU) |

## What I Learned

This project taught me a lot about how CNNs actually work in practice. The bigger takeaways were how much data augmentation helps with overfitting, how batch normalization speeds up training, and how learning rate scheduling can squeeze out a few extra percentage points of accuracy. If I were to improve this further, I'd try transfer learning with a pretrained model like ResNet or EfficientNet, which would probably push accuracy above 95%.
