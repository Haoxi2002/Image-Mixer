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
The required data files should be put into data folder `./data/ECW/`. 
A demo slice of the ECW data is illustrated in the following figure.

![](https://private-user-images.githubusercontent.com/45703329/242822175-5b7189dd-71f0-4097-b945-bdbf51ef43aa.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzI2Njc2NTUsIm5iZiI6MTczMjY2NzM1NSwicGF0aCI6Ii80NTcwMzMyOS8yNDI4MjIxNzUtNWI3MTg5ZGQtNzFmMC00MDk3LWI5NDUtYmRiZjUxZWY0M2FhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDExMjclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMTI3VDAwMjkxNVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWIwZmY2NzFhY2IxZjIyZDAwNmE2YjJjNDQ1MGM2ZDczN2Y2ODRiODQ1NzUxMjQxZDY3ZTY2NmRhZGVjN2ZlZGQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mxcJ-B1JGp4D2zSl5mhsyBEYej471nPDtrJvNaoMggo)

We only use the `bw_upload` columns.

### Train&Test

You can run with the following command for train:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_08.csv --is_training 1 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`

and the following command for test:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_newapp.csv --is_training 0 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`
