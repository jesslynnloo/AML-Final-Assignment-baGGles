# AML-Final-Assignment

This is the Final Assignment for WOA7015 Advanced Machine Learning course, University of Malaya

This assignment is about Autism Spectrum Disorders (ASD) Classification, specifically in hand flapping characteristics classification.


Files:
1. extract.py: The code used to extract frames from videos.
2. resize_normalize.py: The code used to perform resizing and normalization of the frames.
3. Augment.py: The code used to perform data augmentation.
4. main.py: The code used to get the new, unseen input video, and output a final prediction for the input video. Uses extract.py, resize_normalize.py, and post-modelling.py.
5. post-modelling.py: The code for post-modelling pipeline.


In the 'Model training and testing' folder, there are six python notebooks for our experiments, model training and testing.
1. AML_BRNN.ipynb: The model trained is Bidirectional_RNN, and Cross Entropy Loss is used.
2. AML_LSTM.ipynb: The model trained is LSTM, and Cross-Entropy Loss is used.
3. AML_v7_Resnet101.ipynb: The model trained is Resnet-101 based CNN-LSTM, and Cross-Entropy Loss is used.
4. AML_v7_VGG_val.ipynb: The model trained is VGG-16 based CNN-LSTM, and Cross-Entropy Loss is used.
5. AML_v8_Resnet101.ipynb: The model trained is Resnet-101 based CNN-LSTM, and Focal Loss is used.
6. AML_v8_VGG_Focal_Loss.ipynb: The model trained is VGG-16 based CNN-LSTM, and Focal Loss is used.

Model architectures and hyperparameters were tuned manually.


The best model is VGG-16 based CNN-LSTM, with learning rate of 0.001, 30 epochs, and Cross-Entropy Loss is used as the loss function.
For the best model, you can download the .pth file from here: https://drive.google.com/file/d/1-3nzg38NvWVbu5phvDCNiBeNY73WUFM4/view?usp=sharing

The preprocessed dataset used for model training can be accessed here: https://drive.google.com/file/d/12JDTbiIyrzy3TWMU8QDKQ-voek2bgLki/view?usp=sharing
