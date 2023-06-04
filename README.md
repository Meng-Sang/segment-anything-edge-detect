# Segment Anything Edge Detection

This project is a replication of edge detection implemented using Segment Anything. It is reproduced as described in the [Segment Anything paper](https://ai.facebook.com/research/publications/segment-anything/) and [Segment Anything issues#226](https://github.com/facebookresearch/segment-anything/issues/226). Please advise us if there are any problems.


## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install Segment Anything:

```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Install opencv:

```shell
pip install opencv-python
```

## Run

- download weight

  please refer to [segment anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) that is a awesome work.

- edge detect

```shell
python edge_detect.py --edge_dir EDGE_DIR --save_dir SAVE_DIR [--sam_checkpoint SAM_CHECKPOINT] [--model_type MODEL_TYPE] [--device DEVICE]
```



## Demo

<p float="left">
  <img src="assets/1.jpg?raw=true" width="49.1%" />
  <img src="assets/ed_1.jpg?raw=true" width="48.9%" />
</p>

## References

- [Segment Anything project](https://github.com/facebookresearch/segment-anything) 
- [Segment Anything paper](https://ai.facebook.com/research/publications/segment-anything/)
