# ML Image Classifier

CNN image classifier I trained on the CIFAR-10 dataset using TensorFlow/Keras. Classifies images into 10 categories -- airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

I wanted to go beyond just training a basic model, so I added data augmentation, batch normalization, dropout, and learning rate scheduling. Ended up hitting about 92% accuracy on the validation set.

## How it works

The model is a CNN with three conv blocks (each has two Conv2D layers, batch norm, max pooling, and dropout), then flattens into a couple dense layers before the softmax output. Nothing groundbreaking but it works well for this dataset.

Data augmentation made the biggest difference -- random rotations, shifts, and horizontal flips during training helped a lot with overfitting. Without it, the model started memorizing the training data after about 30 epochs.

## Running it

```bash
pip install -r requirements.txt

# train the model
python train.py

# evaluate on test set
python evaluate.py

# classify a single image
python predict.py --image path/to/image.jpg
```

Training takes about 20 minutes with a GPU. The model gets saved automatically and `evaluate.py` will spit out a confusion matrix, per-class accuracy, and a classification report.

## Results

Got about 91.8% test accuracy overall. Ships and trucks did the best (~95%), cats did the worst (~85%) which makes sense -- cats and dogs look pretty similar at 32x32 pixels.

## Files

- `train.py` -- model architecture + training loop with callbacks
- `predict.py` -- classify new images with a saved model
- `evaluate.py` -- full evaluation with confusion matrix and metrics
- `utils/data_loader.py` -- data loading and preprocessing
- `utils/visualization.py` -- plotting functions
- `config.py` -- hyperparameters and paths

## What I'd do differently

If I revisit this I'd probably try transfer learning with something like ResNet or EfficientNet. Would likely push accuracy above 95% without much extra work.
