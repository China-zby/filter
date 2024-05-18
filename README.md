# filter
## install
- using conda(optional)
````
conda create -n baseline python=3.9
conda activate baseline
````
- install independencies
````
pip install -r requirements.txt
````
- run
````
cd filter
python3 baseline.py
````
修改baseline中video的路径为你的视频路径，folder_path修改为你的查询图像文件夹的路径，threshold可以根据实际过滤效果自行调整，这里默认为40
