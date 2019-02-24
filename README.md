# GAN-Pytorch-Template
A Generative Adversarial Networks(GAN) project template to simplify building and training deep learning models using pytorch.  
This repo is designed for those who want to start their projects of GAN. It provides fast experiment setup and attempts to maximize the number of projects killed within the given time. You can build your own GAN easily.

# Table Of Contents

-  [In Details](#in-details)
    -  [Project architecture](#project-architecture)
    -  [Folder structure](#folder-structure)
    -  [ Main Components](#main-components)
        -  [Models](#models)
        -  [Trainer](#trainer)
        -  [Data Loader](#data-loader)
        -  [Logger](#logger)
        -  [Configuration](#configuration)
        -  [train.py](#train.py)
        -  [Inference](#inference)
 -  [Future Work](#future-work)
 -  [Contributing](#contributing)
 -  [Acknowledgments](#acknowledgments)


# In Details

Folder structure
--------------

```
├──  trainers
│   ├── base_model.py        - this file contains the abstract class of the model.
│   ├── base_trainer.py      - this file contains the abstract class of the trainer.
│   ├── example_model.py     - this file contains any model of your project.
│   └── example_trainer.py   - this file contains trainers of your project.
│
│
├── nets                     - this folder contains any net of your project.
│   └── example_net.py
│
│
├── inference.py             - here's the inference of your project.
│   
│   
├── train.py                 - here's the main(s) of your project (you may need more than one main).
│    
│  
├── configs
│    └── config.py           - configuration
│  
│  
├── data_loader  
│    └── dataset.py          - here's the data_generator that is responsible for all data handling.
│ 
└── utils
     ├── logger.py
     ├── utils.py
     └── any_other_utils_you_need

```


## Main Components

### Trainers
--------------
- #### **Base model**
    
    Base model is an abstract class that must be Inherited by any model you create, the idea behind this is that there's much shared stuff between all models.
    The base model contains:
    - ***Save*** -This function to save a checkpoint to the desk. 
    - ***Load*** -This function to load a checkpoint from the desk.
    - ***Cur_epoch*** -These variables to keep track of the current epoch.
    - ***create_model*** Here's an abstract function to define the model, ***Note***: override this function in the model you want to implement.
    - 
- #### **Your model**
    Here's where you implement your model.
    So you should :
    - Create your model class and inherit the base_model class
    - override "create_model" where you write the pytorch net you want
    - 
- #### **Base trainer**
    Base trainer is an abstract class that just wrap the training process.
    
- #### **Your trainer**
     Here's what you should implement in your trainer.
    1. Create your trainer class and inherit the base_trainer class.
    2. override these two functions "train_step", "train_epoch" where you implement the training process of each step and each epoch.

### Data Loader
This class is responsible for all data handling and processing and provide an easy interface that can be used by the trainer.

### Logger
This class is responsible for printer and log-writer.


### Configuration
I use dictionary as configuration method and then parse it, so write all configs you want then parse it using "utils/configs/config.py" and pass this configuration object to all other objects.

### train.py
Here's where you combine all previous part.
1. Parse the config file.
2. Create a tensorflow session.
2. Create an instance of "Model", "Dataset" and "Logger" and parse the config to all of them.
3. Create an instance of "Trainer" and pass all previous objects to it.
4. Now you can train your model by calling "Trainer.train()"

### inference
Here's where you translate images.

# References
1.[https://github.com/pytorch](https://github.com/pytorch)  
2.[https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)  
3.[https://pytorch.org](https://pytorch.org)  
4.[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
5.[https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)  
6.[https://github.com/CDOTAD/AlphaGAN-Matting](https://github.com/CDOTAD/AlphaGAN-Matting)  

