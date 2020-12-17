# MobileFaceNet-Pruning

## 1. Installing Requirements
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt    
```

## 2. Preparing Dataset
1. Install the following dataset   
    * [Faces_ms1m-refine-v2](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
2. Set datapath in `parser.py`
3. Run `python3 prepare_data.py`

## 3. Training
1. Set training settings and datapath in `parser.py`
2. Run `python3 main.py --options train`

## 4. Pruning
1. Run `python3 main.py --options prune`

- - -
## 5. References
* [Pruning_MTCNN_MobileFaceNet_Using_Pytorch](https://github.com/xuexingyu24/Pruning_MTCNN_MobileFaceNet_Using_Pytorch)
* [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
* [MobileFaceNet_TF](https://github.com/sirius-ai/MobileFaceNet_TF)

