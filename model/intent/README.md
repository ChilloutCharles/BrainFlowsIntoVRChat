# Intent Classification
This folder contains the code to generate a model to attempt to guess between two thoughts (baseline and fireball). To use this folder, make sure to rerun Step 5 from the main readme to get the new depenencies.

## Pipeline
1. Detrend and denoise: Remove over time trendlines and remove 50, 60hz noise.
2. Bandpass to gamma and high gamma for frequencies associated with higher thought
3. Perform Wavelet Transform on filtered channels, stacking them into a 2D feature vector
4. Classify 2D feature vector against a Convolutional Neural Network

## Recording
Recording eeg is done by doing a 30 second baseline session and three 10 seconds sessions. Each session will ask you to think a specific thought and will start once you press the enter button.

Command to start the recording session: `python record_eeg.py --board-id <YOUR BOARD ID>`

A new file `recorded_eeg.pkl` will be generated containing the session data.

## Training
Once the recording session is done, training can start. This involves: 
- generate 1 second windows from the session data.
- split the windows to train and validation sets.
- preprocess and extract 2D features.
- train CNN model
- validate trained model against the validation set.

Command to start training: `python train.py`

A new file `shallow.keras` will be generated, containing the CNN model.

### Notes
Baseline Accuracy - You will want to keep baseline as accurate as possible, since a false positive is worse than a false negative. Make sure that `1 recall` is at least comparable to the `f1-score accuracy`. If not, the model needs to be retrained.

## Testing
A test script is added to test out how well the model behaves in real time. The results will be displayed at 60hz refresh rate with an ema_decay of 1/60.

Command to start testing: `python test.py --board-id <YOUR BOARD ID>`

## Considerations

I am blown away by how effective using Convolutional Neural Networks are with this. Wavelet transformations decompose signals into components that capture both frequency and time information. Stacking these rows of components from each channel creates something like an "image" where the "color" is determined by the component values. It's this spatial-like data that CNNs excel at understanding, finding frequency patterns over time and channel location.

There is still work to be done in terms of thought categorization. While the model performs fairly well, isn't generalized enough to differentiate between other thoughts that required active thinking and in the future will need more training data (ex. fireball vs waterball).

Training sessions have been redone to get a better baseline.

## Usage with BFiVRC

To use the model within VRChat, redo the depndency step, go through recording and training, then add the launch argument `--enable-intent` when running main

The avatar variable name is `BFI/MLIntent/Action` which will give back a float of range [0, 1], which corresponds to the predicted probability you're thinking an action.
