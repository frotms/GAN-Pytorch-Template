# coding=utf-8
import os
import json

global_config = {
  "gpu_id": "0,1,4,7,9,15",
  "async_loading": true,
  "shuffle": true,
  "data_aug": true,

  "num_epochs": 3000,
  "img_height": 320,
  "img_width": 320,
  "num_channels": 3,

  "batch_size": 96,
  "dataloader_workers": 1,
  "learning_rate_g": 1e-4,
  "learning_rate_decay_g": 0.9,
  "learning_rate_decay_epoch_g": 100,
  "learning_rate_d": 1e-4,
  "learning_rate_decay_d": 0.9,
  "learning_rate_decay_epoch_d": 100,
  "every_d": 3,
  "every_g": 1,

  "save_path": "your_save_path",
  "save_name": "model.pth",

}

if __name__ == '__main__':
    config = global_config
    print(config['gpu_id'])
    print('done')