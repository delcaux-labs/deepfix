---
name: model-training
description: Instructions on how to use the provided training logic to train and evaluate vision models (like image classification models) using deepfix-server's model-training skill.
---
# Model Training Skill

This skill provides baseline training scripts to give you ideas on how to use PyTorch Lightning and timm models for image classification. You can use these as a starting point or inspiration for your own training code.

## Provided Files

- `classification.py`: Contains `ClassificationTrainerConfig`, `ClassifierModule`, `ClassificationDataModule`, and `ClassificationTrainer`.
- `timm_models.py`: Contains a timm-backed classification model, `TimmClassificationModel`, and related utilities.

## Usage

When you need to perform training (e.g., training a classifier on a vision dataset), you should import the components from this skill directory (i.e. `deepfix_server.skills.model_training`).

Example of using the `ClassificationTrainer`:

```python
import os
import sys

# Ensure you can import from the skill
sys.path.insert(0, os.path.abspath("deepfix-server/src"))

from deepfix_server.skills.model_training.classification import (
    ClassificationTrainer, 
    ClassificationTrainerConfig
)
from deepfix_server.skills.model_training.timm_models import TimmClassificationModel

config = ClassificationTrainerConfig(
    num_classes=10,
    epochs=5,
    # other configurations...
)

model = TimmClassificationModel(
    timm_model_name="vit_small_patch16_224.dino",
    labels_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
)

trainer = ClassificationTrainer(config)
trainer.run(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset
)
```

**Note**: Be sure to prepare your `train_dataset` and `val_dataset` appropriately as PyTorch `Dataset` instances before calling `trainer.run()`.
