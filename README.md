## N2V: An Image-Driven Lightweight Model for Network Key Indicators Extreme Forecasting

This is the origin Pytorch implementation of N2V.

### Requirements

- torch==1.11.0
- numpy==1.26.4
- reformer-pytorch==1.4.4
- pandas==1.5.3
- matplotlib
- scikit-learn
- einops

Dependencies can be installed using the following command:

`pip install -r requirements.txt`

### Data

The ECW dataset used in the paper can be downloaded in the repo [ECWDataset](https://github.com/hsy23/ECWDataset). 
The required data files has already been put into data folder `./data/ECW/`. 

We only use the `bw_upload` columns.

### Train&Test

You can run with the following command for train:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_08.csv --is_training 1 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`

and the following command for test:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_newapp.csv --is_training 0 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`
