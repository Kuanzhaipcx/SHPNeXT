# SHPNeXT
SHPNeXT: Enhanced Tongue Image Segmentation Across Multi-scale and Variable Resolutions for Traditional Chinese Medicine

## 1.Preparation
The project requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.
**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 3.** The method code proposed in this article are ``` SHPNeXt_config.py ```, ``` SHPNeXt_backbone.py ```, ```poolformer.py```, ```hire-mlp.py``` and ```ham_head_nuclearnmf_decoder.py ```. 
Put the ``` SHPNeXt_backbone.py ```, ```poolformer.py``` and ```hire-mlp.py``` into ```mmseg/models/backbones```, put the ```ham_head_nuclearnmf_decoder.py ``` into ```mmseg/models/decode_heads```, and put the ``` SHPNeXt_config.py ``` into dir of ```config```.

**Step 4.** Prepare the datasets. The two open source data sets ```BioHit``` and ```LRCM ``` used in this article can download from the link [Dataset](https://drive.google.com/file/d/1CTfb5x9I79FUreqRgssu01hqnsjh8h6i/view?usp=drive_link). Due to privacy restrictions, we are unable to provide the ```HUCM ``` data set.

## 2.Installation

**Note:**
Our job based on the Project of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).We recommend the users install the relative packges.

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install MMSegmentation.

Case a: If you develop and run mmseg directly, install it from source:

```shell
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

Case b: If you use mmsegmentation as a dependency or third-party package, install it with pip:

```shell
pip install "mmsegmentation>=1.0.0"
```
 Follow the mmsegmentation tutorial. Register and configure settings and you're ready to run.

## 4.Train
We provide `tools/train.py` to launch training jobs on a single GPU.
You can train the all the models presented in the paper by use the following command.

```shell
python tools/train.py  ${CONFIG_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--amp`: Use auto mixed precision training.
- `--resume`: Resume from the latest checkpoint in the work_dir automatically.
- `--cfg-options ${OVERRIDE_CONFIGS}`: Override some settings in the used config, and the key-value pair in xxx=yyy format will be merged into the config file.

For example, if you want train the ```SHPNeXt```, run
```shell
python tools/train.py  configs/SHPNeXt_config.py --work-dir /path/your_choice
```

## 5.Test
You can directly use our trained model（Download the weights: [SHPNext_BioHit](https://drive.google.com/file/d/1UkE1BgS3VwGDdCto3VLYc8uFngLEEZvM/view?usp=drive_link), [SHPNext_LRCM](https://drive.google.com/file/d/1couq9_X8Wvcg-rmAR3JoWP82Qz0FZDbY/view?usp=drive_link)） files for quick testing, or use your trained model for testing. Use the following command:

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

This tool accepts several optional arguments, including:

- `--work-dir`: If specified, results will be saved in this directory. If not specified, the results will be automatically saved to `work_dirs/{CONFIG_NAME}`.
- `--show`: Show prediction results at runtime, available when `--show-dir` is not specified.
- `--show-dir`: Directory where painted images will be saved. If specified, the visualized segmentation mask will be saved to the `work_dir/timestamp/show_dir`.
- `--wait-time`: The interval of show (s), which takes effect when `--show` is activated. Default to 2.
- `--cfg-options`:  If specified, the key-value pair in xxx=yyy format will be merged into the config file.
- `--tta`: Test time augmentation option.

For example, if you want test the ```SHPNeXt``` using our trained model, run
```shell
python tools/train.py  configs/SHPNeXt_config.py SHPNext_BioHit.pth
```

## 6.Contact
If you have any questions, please feel free to contact: ```pengchongxiaoKZ@163.com```
