# Usage

**Required dependencies**: PyTorch, pandas, numpy, tqdm, skimage

Execute "Regression.py" to train the network and see loss function values.

**Command line options**:

- -s, --save-model --> if present this flag forces saving of trained model
- --learning-rate, --batch-size, --epochs --> to specify parameters to use, defaults are 0.001, 128, 20
- -p --> plot predicted vs original labels
- --time-frame --> daily, weekly, monthly, seasonally, deafults to daily
