# Covid-19-Detection-From_XRAY
## Dataset is taken from https://github.com/ieee8023/covid-chestxray-dataset

Dataset Description: View: PA
> COVID:
>> ieee8023.github Dataset: 123
> Non COVID:
>> ieee8023.github Dataset: 46 

>>> ARDS: 5 

>>> Bacterial Pneumonia: 17 

>>> Chlamydophila (Bacterial): 1 

>>> Fungal Pneumonia: 13 

>>>Klebsiella (Bacterial): 1 

>>>Legionella (Bacterial): 2 

>>>MERS (CORONA Virus): 0 

>>>No Finding: 1 

>>>Pneumocystis (Fungal): 13 

>>>SARS (CORONA Virus): 11 

>>>Streptococcus (Bacterial): 13 

>>>Viral Pneumonia: 110

# Kaggle Pneumonia Dataset: 72
> Normal: 33
> Pneumonia: 39
Model Description:
## InceptionResNetV2 -> AveragePooling -> Flatten -> Dense -> Dense(Result)

# Description:
### Loss Function: Cosine Similarity
### Pretrained weight: ImageNet(layers are trainable)
### Optimizer: sgd
### lr: 0.001
### Batch size: 10
### Epoch: 50
### Early Stop: True (after 20 epoch(patience))
### Decay rate: lr/Epoch
### Result: 10 Fold Cross-validation

## Fold no: 1

[Op:__inference_distributed_function_586364]
Function call stack:
distributed_function
Optimal Average Accuracy: 92.22%
Optimal Highest Accuracy: 100%
Trivial Highest Accuracy: 95.83%
Ill Doing:
Did not save model after every epoch
Possible Solution:
1. Discard SARS and MERS data from Non Covid dataset to increase noncovid
accuracy
2. Brightness augmentation and rotation augmentation should not be
applied. Rotational augmentation also should not be applied 150%
Zooming augmentation applied.
