# cv-face-mask-detection

## Contributors

- Tien Nguyen: MS Data Science at Drexel University
- Khiem Nguyen: MS Data Science at Drexel University

## Install packages
```
pip install -r requirements.txt
```
or 
```
pip3 install -r requirements.txt
```

## Dataset
Download the dataset at https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view 

Move the dataset to the **dataset** folder and run **processing.py** to process the data. Detailed command as follow:
```
cd dataset/
python3 processing.py
```
This processing step will extract facial features from the dataset. See example below:

### *Before processing*
![Before extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/handshaking.jpg)

### *After processing*
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_148_158_202_276.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_348_86_402_214.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_388_190_448_264.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_764_124_820_236.jpg)

## Training
Training files has the format: "train_{model}.py". Where model:
  - pretrained: Fine-tuning a pretrained AlexNet model
  - alexnet: training a AlexNet model
  - resnet: training a ResNet model
  - mobile: training a MobileNetV2 model

To run the training, execute:
```
python3 [training file]
```

## Training on Google Colab
Export the train_gg_colab.ipynb to Google Colab to start training on Google Colab 

## Execution
**face_mask_detect.py**: to run the Face Mask Dectection module.
