# EmoAudioNet
Here the code of EmoAudioNet is a deep neural network for speech classification (published in ICPR 2020). 

# Authors
The code is writing by Kamil Bentounes and Daoud Kadoch under the supervision of Dr. Alice OTHMANI

# Citation
If you want to use this code, thanks for citing:

Othmani, A., Kadoch, D., Bentounes, K., Rejaibi, E., Alfred, R., and Hadid,A. (2020). Towards robust deep neural networks for affect and depression recognition. ICPR 2020.

# Licence
This project is licensed under the MIT License - see the LICENSE.md file for details

## Installation

Here you can install all dependencies required for this project with: 

```bash
python -m pip install -r requirements.txt
```

## Usage

After installing all dependencies:

* If you have a small dataset and you need data augmentation step, download your dataset and put it on the root of the project folder. Make sure that all folders contain your audio files and decomment only `augmentation()` before run `python spectro_dir.py`. 

* To generate spectrograms, decomment only `create_spectro_dir(DELETE=True/false)` and run `python spectro_dir.py`.

* To create labels csv, decomment only `create_csv('labels.csv')` and run `python spectro_dir.py`.

* To resize and crop all spectrograms according to input CNN Spectorgrams based by decommenting only `resize(x, y)` and run `python spectro_dir.py`. Then run `python crop.py`.

* To generate MFCC data, you must run `python DATA_LOAD.py` to read labels.csv (which must be on the same root). This script generates two files: `SPECTRO.pkl` and `MFCC.pkl` which contain all features we need according to the input model.

* To train the model, you should run `python Concatenate_Model.py`. You must have generated all required features (MFCC and Spectrograms) with pickle. 

