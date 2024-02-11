# Intent Classification
This folder contains the code to generate a model to attempt to guess between two thoughts (baseline and fireball). To use this folder, make sure to rerun Step 5 from the main readme to get the new depenencies.

## Pipeline
1. Detrend and denoise: Remove over time trendlines and remove 50, 60hz noise.
2. Bandpass to gamma frequencies associated with higher thought.
3. Perform Wavelet Transform on filtered signal and treat wavelet coefficients as a feature vector.
4. Scale the features against the normal distribution and perform Principal Component Analysis to reduce the dimensions of the feature vector while retaining 95% variance.
5. Classify the reduced feature vector using a Support Vector Machine

## Recording
Recording eeg is done by doing six 10 second sessions. Each session will ask you to think a specific thought and will start once you press the enter button.

Command to start the recording session: `python record_eeg.py --board-id <YOUR BOARD ID>`

A new file `recorded_eeg.pkl` will be generated containing the session data.

## Training
Once the recording session is done, training can start. This involves: 
- generate 1 second windows from the session data.
- split the windows to train and validation sets.
- fit the scaler and PCA models against the training set.
- train and tune the SVM model.
- validate trained pipeline against the validation set.

Command to start training: `python train.py`

A new file `models.ml` will be generated, containing the scaler, PCA, and SVM models.

### Notes
Overfitting - Compare the `Train Score` vs the `f1-score accuracy`. If the `Train Score` is high but `f1-score accuracy` is not, the model overfitted against the training data and will need to be retrained.

Baseline Accuracy - You will want to keep baseline as accurate as possible, since a false positive is worse than a false negative. Make sure that `baseline recall` is at least comparable to the `f1-score accuracy`. If not, the model needs to be retrained.

## Testing
A test script is added to test out how well the model behaves in real time. The results will be displayed at 60hz refresh rate with an ema_decay of 1/60.

Command to start testing: `python test.py --board-id <YOUR BOARD ID>`

## Considerations
From what I found testing this myself, the trained model is not generalized enough to hold over long periods of time. Seems that it cannot handle the data drift after testing the model days after the recording was taken.

There is still work to be done in terms of thought categorization as well. The model isn't generalized enough to differentiate between other thoughts that required active thinking (ex. fireball vs waterball).

The model is also sensitive on the headband you are using. If I switch between my muse 2, which I trained on, to muse 1, it will no longer trigger outside of baseline.

## Usage with BFiVRC

To use the model within VRChat, go through recording and training, then add the launch argument `--enable-intent` when running main

The avatar variable name is `BFI\MLIntent\Action` which will give back a float of range [0, 1], which corresponds to the predicted probability you're thinking an action.
