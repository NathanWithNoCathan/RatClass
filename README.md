# RatClass

RatClass is a small project to analyze which popular image classification models and configurations perform best on small datasets. RatClass tries to classify different visual characteristics of pet rats, such as coat color and other genetic traits, using a dataset of images of fancy rats. The goal is to evaluate how different model architectures, sizes, learning rates, backbone freezing strategies, and augmentation techniques affect performance on a small dataset of around 50-150 images per class.

## Models Evaluated

RatClass uses 4 different models for evaluation: ResNet, MobileNet, EfficientNet, and DenseNet. Different sizes of each model are tested to determine which architecture and size perform best on the given dataset.

In addition to varying the model, some other parameters related to small dataset training are also varied, including learning rates, backbone freezing strategies, and augmentation techniques. This allows for a comprehensive analysis of how different configurations affect model performance on limited data.

This is a refined version of the original plan, which included significantly more models and configurations that may not be feasible to run in a reasonable timeframe, or may not have given meaningful insights.

In total:

- 4 models (ResNet, MobileNet, EfficientNet, DenseNet)
- 2 sizes for each model (small, medium)
- 2 learning rates (1e-4, 3e-3)
- 2 different backbone freezing strategies (freeze backbone for first 5 epochs, or freeze backbone for first 15 epochs)
- 6 different augmentation techniques:
    - Minimal
    - Basic geometric
    - Strong geometric
    - Geometric + color
    - Strong geometric + color
    - Strong geometric + color + random erasing

This results in 4 x 2 x 2 x 2 x 6 = 192 different training runs to evaluate the performance of each model under various configurations.

I decided to make many factors between training runs constant in order to actually evaluate the factors being varied appropriately. Here are a list of the factors that are kept constant across all training runs:

- Pretrained weights (all models are initialized with pretrained weights)
- All models will use the AdamW optimizer with the same weight decay (1e-4) and no momentum, to avoid confounding the results with different optimizers or hyperparameters.
- Cosine annealing learning rate scheduler
- Batch size of 16
- Max epochs of 40, with early stopping after 10 epochs of no improvement in validation loss.

## Dataset

The dataset is compiled from images shared by multiple breeders across the United States. Permission was explicitly obtained by those breeders. Specific breeder credits will be listed in the paper acknowledgements, and are listed in a section below. 

The data is organized by class first, with most nested folders corresponding to individual rats that have been classified into a larger coat-based category. For example, Mahina, Moana, and Akala are all Martens, and are in the marten class. In addition, each class contains an assortment folder that groups together rats from that class when only 1 or 2 pictures are available for a given individual, avoiding the need to create a separate folder for every sparse example. The dataset is small, with only 50-100 images per class, which makes it a good test case for evaluating model performance on limited data.

The dataset loader in [load_dataset.py](load_dataset.py) uses `torchvision.datasets.ImageFolder`, so each top-level folder in `./dataset` becomes a class label automatically, while the nested folders are scanned recursively for images regardless of whether they correspond to one rat or the class-level assortment folder for sparse examples.

### Note on Class Labels

The class labels given to the model are based on visible coat characteristics, such as color and variations. These characteristics are often determined by specific genetic traits, but those traits are not necessarily mutually exclusive to the other traits. For example, a rat can simultaneously be a marten and a black, but the model is only given the marten label, since the marten coat pattern is more visually distinctive than the black coat color. This means that the model is not necessarily learning to identify specific genetic traits, but rather learning to classify based on visual characteristics that may be influenced by multiple underlying traits.

This project is not intended to perfectly classify all the different traits of a rat, but rather to evaluate how different model architectures and configurations perform on a small dataset with visually distinctive classes. However, future work may involve exploring multi-label classification to allow the model to learn to identify multiple traits simultaneously, which could potentially provide more insights into the underlying genetics of coat patterns in rats.

### Dataset Limitations

There are a number of limitations to this dataset that should be noted:

- A large portion of the dataset is made up of juvenile rats, which may have different coat patterns and colors than adult rats. This could potentially affect the model's ability to generalize to adult rats.
- The dataset is relatively small, with only 50-150 images per class. This may limit the model's ability to learn complex features and could lead to overfitting. That being said, the small dataset size is intentional for this project, as the goal is to evaluate model performance on limited data.
- The dataset is imbalanced, with some classes having more images than others. This could potentially bias the model towards the more represented classes and affect its performance on the less represented classes.

### Dataset Credits

A big thank you to the breeders who allowed me to use their images for this project. If you are interested in getting pet rats, please consider supporting these breeders and giving them credit for their work in caring for and sharing these wonderful animals.

- [Kolohe Iole Rattery](https://koloheiolerattery.com/) - Oahu, Hawaii (ships to other islands)
- [Little Paws Rattery](https://iowalittlepawsrattery.weebly.com/) - Osceola, Iowa

## Technologies used

- Python
- PyTorch
- Torchvision
- OpenCV (for data augmentation)
- Matplotlib (for visualizations)

## Setup

Install the core dependencies in your virtual environment:

```bash
pip install torch torchvision opencv-python matplotlib
```

I used Python 3.12 for this project, but it may work with other versions as well.