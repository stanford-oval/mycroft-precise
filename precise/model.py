# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import attr
from os.path import isfile
from typing import *

from precise.functions import load_keras, false_pos, false_neg, f_score, weighted_log_loss, set_loss_bias
from precise.params import inject_params, pr

if TYPE_CHECKING:
    from keras.models import Sequential


@attr.s()
class ModelParams:
    """
    Attributes:
        recurrent_units:
        dropout:
        extra_metrics: Whether to include false positive and false negative metrics
        skip_acc: Whether to skip accuracy calculation while training
    """
    recurrent_units = attr.ib(50)  # type: int
    input_dropout = attr.ib(0.2)  # type: float
    dropout = attr.ib(0.4) # type: float
    weight_decay = attr.ib(1e-4) # type: float
    extra_metrics = attr.ib(False)  # type: bool
    skip_acc = attr.ib(False)  # type: bool
    loss_bias = attr.ib(0.7)  # type: float
    freeze_till = attr.ib(0)  # type: bool


def load_precise_model(model_name: str) -> Any:
    """Loads a Keras model from file, handling custom loss function"""
    if not model_name.endswith('.net'):
        print('Warning: Unknown model type, ', model_name)

    inject_params(model_name)
    return load_keras().models.load_model(model_name)


def create_model(model_name: Optional[str], params: ModelParams) -> 'Sequential':
    """
    Load or create a precise model

    Args:
        model_name: Name of model
        params: Parameters used to create the model

    Returns:
        model: Loaded Keras model
    """
    if model_name and isfile(model_name):
        print('Loading from ' + model_name + '...')
        model = load_precise_model(model_name)
    else:
        from keras.layers.core import Dense, Dropout
        from keras.layers.recurrent import GRU
        from keras.models import Sequential
        from keras import regularizers

        l2 = regularizers.l2(params.weight_decay)

        model = Sequential()
        model.add(GRU(
            params.recurrent_units, activation='tanh',
            input_shape=(pr.n_features, pr.feature_size), dropout=params.input_dropout, name='gru0', unroll=True
        ))
        model.add(Dropout(params.dropout))
        model.add(Dense(params.recurrent_units,
                        kernel_regularizer=l2,
                        activation='relu'))
        model.add(Dropout(params.dropout))
        model.add(Dense(1,
                        kernel_regularizer=l2,
                        activation='sigmoid'))

    load_keras()
    metrics = ['accuracy', f_score] + params.extra_metrics * [false_pos, false_neg]
    set_loss_bias(params.loss_bias)
    for i in model.layers[:params.freeze_till]:
        i.trainable = False
    from keras import optimizers
    
    sgd = optimizers.SGD(lr=1.0, decay=1e-4)
    model.compile(optimizer=sgd, loss=weighted_log_loss, metrics=(not params.skip_acc) * metrics)
    return model
