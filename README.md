# cv-face-mask-detection
*Please do not move the files*

## Dataset
Download the dataset at https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view 

Move the dataset to the **dataset** folder and run **processing.py** to process the data. Detailed command as follow:
```
cd dataset/
python3 processing.py
```
This processing step will extract facial features from the dataset. See example below:

### Before
![Before extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113.jpg)

### After
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_148_158_202_276.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_348_86_402_214.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_388_190_448_264.jpg)
![After extracting](https://raw.githubusercontent.com/ngkhiem97/cv-face-mask-detection/main/images/1_Handshaking_Handshaking_1_113_764_124_820_236.jpg)

## Training
To train the model, run the **train.py** file as below:
```
python3 train.py
```

## Execution
1. **face_mask_detect.py**: to run the Face Mask Dectection module.
2. **predict.py**: to predict an individual image.
