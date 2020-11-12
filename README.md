# AEDNet  
This repo contains the Pytorch implementation of the ICCST2020 paper - Video Summarization with Self-Attention Based Encoder-Decoder Framework  
![image]("https://github.com/pingxvyufeng/AEDNet/blob/main/image/model.jpg"/)    
The main requirements are pytorch (v1.6.0) and python 3.6. Some dependencies that may not be installed in your machine are tabulate and h5py. Please install other missing dependencies.
## Installation  
The development and evaluation was done on the following configuration:  
### System configuration
* Platform :Ubuntu 18.04.4 LTS
* Display driver : NVIDIA-SMI 450.51.05
* GPU: NVIDIA 2070 super
* CUDA: 9.0.176
* CUDNN: 7.1.2
### Python packages
* Python: 3.6.0
* PyTorch: 1.6.0
* numpy 1.19.1
* ortools: 6.9.5824
## Get started
Preprocessed datasets TVSum, SumMe, YouTube and OVP as well as VASNet pretrained models you can download by running the following command:
```
./download.sh datasets_models_urls.txt
```
You will need about 820MB space on your HDD. Datasets will be stored in ./datasets directory and models, with corresponding split files, in ./data/models and ./data/splits respectively.
It may be difficult for Chinese developers to download the training data set, so I will provide the data link of Baidu Netdisk here to download the training data.  
Link:https://pan.baidu.com/s/1Sz1sR3woO50TL6K0XqWO4Q
Code:l81m  
## Evaluation
To evaluate all splits in ./data/splits with corresponding trained models in ./data/models run the following:
```
python3 main.py
```
## Training
To train the AEDNet on all split files in the ./splits directory run this command:
```
python3 main.py -- train
```
The final result will be recorded in the ./data/models directory along with the corresponding model./data/results.txt.
## Acknowledgement
We would like to thank to K. Zhou et al. and K Zhang et al. for making the preprocessed datasets publicly available and also K. Zhou et al and Jiri Fajtl et al. for code we copied from https://github.com/KaiyangZhou/pytorch-vsumm-reinforce ,https://github.com/ok1zjf/VASNet and slightly modified.
