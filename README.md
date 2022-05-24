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

## Training
To train the model, run the **train.py** file as below:
```
python3 train.py
```

## Execution
1. **face_mask_detect.py**: to run the Face Mask Dectection module.
2. **predict.py**: to predict an individual image.
