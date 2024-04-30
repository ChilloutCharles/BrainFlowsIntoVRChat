# Action Classification
This folder contains the code needed to record, train, and test a machine learning model that will predict a single action from a set of actions from your brainwaves.

## Prerequisites
1. Rerun this command at the base directory to install needed depedencies: `python -m pip install -r requirements.txt`

## Recording your Brainwaves
1. Determine the amount of actions you'd like to record
2. Determine the board id or name of your headband
3. Within this directory, execute this command 
   - `python record_eeg.py --board-id <YOUR BOARD ID> --actions <ACTION COUNT>`
   - optional arguments:
      - `--sessions`       number of sessions per action to record   (Default: 2)
      - `--window-length`  length in seconds of eeg data pulled per session   (Default: 10)
      - `--window-buffer`  time in seconds before eeg data is recorded each session (delay after hitting enter)   (Default: 2)
4. Follow on screen commands to completion. A file named `recorded_eeg.pkl` will be generated.

* You may record multiple training sessions using the `--overwrite 0` argument. This disables overwriting any `.pkl` files, and will instead name them something else (`recorded_eeg1.pkl`, `recorded_eeg2.pkl`, etc.) 
   * This means that you can combine several training sessions worth of data when training the model.
* By default this is set to 1, meaning that any `.pkl` files that exist will be deleted and replaced with the results from that session.

#### Some tips before starting
 - make sure to take care of your surroundings, particularly matching as close as possible to where you would be using this.
 - when thinking of an action, pick something that really stands out to you either visually or conceptually.
 - its highly advised to have one action where you don't do anything at all. This is typically Action 0

## Training the model
1. Execute the command `python train.py`. This training part should take a few minutes.
2. Once the training is done, a classification report and a window showing the error graph over time will be displayed.
   - The graph should show the orange line closely following the blue line smoothly to zero. If it doesn't, either redo step 1 or redo the recording
   - If the classification report shows numbers that aren't all that promising, redo the recording, taking care of the environment you are in.
3. Close the window. A new file containing the model will be created: `shallow.keras`

## Testing the model
1. Execute command `python test.py --board-id <YOUR BOARD ID>`
2. The list of numbers that appear will correspond to actions. The zeroth action will be the first score, first action the second, and so on.
3. The last number will correspond to the action index with the highest score.
4. Test here to see how well it works in realtime. If it doesn't feel satisfactory,either retrain or re-record.

## Usage with BFiVRC
To use the model within VRChat, add the launch argument `--enable-action` when running `main.py`. You can adjust how reactive the model is by adding another launch argument `--action-ema-multiplier <multiplier>`, where `<multiplier>` is defaulted to 5.0
Here are the parameters that will be returned:
- `BFI/MLAction/Action<ID>` (float [0.0, 1.0]) : The score of an action with the index `<ID>`. The higher the score, the higher chance the model thinks you are thinking this action. Example: `BFI/MLAction/Action7`
- `BFI/MLAction/Action` (int) : The action index of the action that has the highest score.
