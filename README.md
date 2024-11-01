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

![](https://private-user-images.githubusercontent.com/45703329/243288877-b523ec9f-0e2f-49d6-9780-687a903790fd.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjQxMjM5OTQsIm5iZiI6MTcyNDEyMzY5NCwicGF0aCI6Ii80NTcwMzMyOS8yNDMyODg4NzctYjUyM2VjOWYtMGUyZi00OWQ2LTk3ODAtNjg3YTkwMzc5MGZkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA4MjAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwODIwVDAzMTQ1NFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMxMjFlY2E2YzQyOGQyY2ZhMWI2Y2ZmOWZjYTBjNDhlMGYyOTk0NzQ1NWY4YjIwNTNlOTdhZWZjZGMwOGUzZDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1sUCLY7bBMJr0Ljrm9psh7SGVPhm-gLao7hw6-t9uj8)

We only use the `bw_upload` columns.

### Train&Test

You can run with the following command for train:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_08.csv --is_training 1 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`

and the following command for test:

`nohup python run.py --data ECW --dir_path ./data/ECW --data_path ECW_newapp.csv --is_training 0 --model Image-Mixer --model_type 0 --batch_size 512 > ECW_Image-Mixer.txt 2>&1 &`