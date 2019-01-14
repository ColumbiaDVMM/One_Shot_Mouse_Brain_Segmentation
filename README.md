# One-Shot Brain Segmentation

Download data (data\_10\_07.zip) from 
[Google Drive](https://drive.google.com/open?id=1xzSIuA2kMq6AMJ4yiduu4S1H-aUZRMg4). Unzip the data to ./data/

Run

```bash
cd code
```

For evaluation of all labeled regions. Feel free to change region_list to test more. 

```python
python run_multi_region_training.py
```

To see the result for 95 regions. It takes about 5 hours for one training setting.

```python
python run_all_region_training.py
```
