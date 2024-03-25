from keras.models import Model, Sequential
from keras.layers import GRU, SeparableConv1D, GlobalAveragePooling1D, BatchNormalization, Dense, Dropout, concatenate
import keras

## Define the model
# cnn branch inspired by thin mobilenet 
# https://scholarworks.iupui.edu/server/api/core/bitstreams/a7fbc815-0f25-480a-bce1-0cb231238b66/content
# adding GRU branch to have temporal details

@keras.saving.register_keras_serializable()
class CNNGRUModel(Model):
    def __init__(self, classes, **kwargs):
        super(CNNGRUModel, self).__init__(**kwargs)
        
        # config to be saved
        self.classes = classes
        
        # CNN branch
        self.cnn = Sequential()
        self.cnn.add(SeparableConv1D(8, 3, activation='relu'))
        self.cnn.add(BatchNormalization())
        self.cnn.add(SeparableConv1D(16, 3, activation='relu'))
        self.cnn.add(BatchNormalization())
        self.cnn.add(SeparableConv1D(32, 3, activation='relu'))
        self.cnn.add(BatchNormalization())
        self.cnn.add(GlobalAveragePooling1D())
        
        # GRU branch
        self.gru = Sequential()
        self.gru.add(GRU(16, return_sequences=True))
        self.gru.add(GRU(32))
        self.gru.add(BatchNormalization())
        
        # Fully connected layer
        self.fc = Sequential()
        self.fc.add(Dropout(0.1))
        self.fc.add(Dense(self.classes, activation='softmax'))
        
    def call(self, inputs):
        x_cnn = self.cnn(inputs)
        x_gru = self.gru(inputs)
        x_combined = concatenate([x_cnn, x_gru], axis=-1)
        return self.fc(x_combined)
    
    # overriden get_config method to save classes count
    def get_config(self):
        config = super().get_config()
        config.update({'classes': self.classes})
        return config