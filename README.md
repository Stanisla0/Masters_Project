# Masters_Project

Prerequisites:
- Python 3.9
- D-Wave Leap account (API key required) https://cloud.dwavesys.com/leap/login/  

How to run code for Wisconsin Breast Cancer (WBC) dataset:
1. export DWAVE_API_TOKEN='user_API_private_key'
2. python MS_Dwave_WBC.py
3. Expected output csv file named: 'results.csv'

How to run code for Surgical-Deepnet dataset:
1. export DWAVE_API_TOKEN='user_API_private_key'
2. Comment out data for fraud_detection_bank_dataset.csv
3. python MS_Dwave_Other.py
4. Expected output csv file named: 'results_2.csv'

How to run code for fraud_detection_bank_dataset dataset:
1. export DWAVE_API_TOKEN='user_API_private_key'
2. Comment out data for Surgical-Deepnet.csv
3. python MS_Dwave_Other.py
4. Expected output csv file named: 'results_2.csv'
