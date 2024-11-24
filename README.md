## 运行本项目
1. Python版本为**3.11.10**。你可以通过conda建立环境
```
conda create -n SNN_ENV python=3.11.10
```

2. 之后进入项目所在路径，在环境内安装所需包。
```
pip install -r .\requirements.txt
```

3. 运行 **main.py**，将自动下载、预处理数据集，并训练测试模型。训练的模型保存在\.\\models_saved中，测试结果图标保存在\.\\figs中。
```
python .\main.py
```
