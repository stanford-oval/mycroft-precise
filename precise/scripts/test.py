#!/usr/bin/env python3
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
from prettyparse import create_parser
import numpy as np

from precise.network_runner import Listener
from precise.params import inject_params, pr
from precise.stats import Stats
from precise.threshold_decoder import ThresholdDecoder
from precise.train_data import TrainData

usage = '''
    Test a model against a dataset
    
    :model str
        Either Keras (.net) or TensorFlow (.pb) model to test
    
    :-u --use-train
        Evaluate training data instead of test data
    
    :-d --use-decoder
        Use ThresholdDecoder
    
    :-nf --no-filenames
        Don't print out the names of files that failed
    
    :-t --threshold float 0.5
        Network output required to be considered an activation
    
    ...
'''


def main():
    args = TrainData.parse_args(create_parser(usage))

    inject_params(args.model)

    data = TrainData.from_both(args.tags_file, args.tags_folder, args.folder)
    train, test = data.load(args.use_train, not args.use_train, shuffle=False)
    inputs, targets = train if args.use_train else test

    filenames = sum(data.train_files if args.use_train else data.test_files, [])
    raw_outputs = Listener.find_runner(args.model)(args.model).predict(inputs)
    
    if args.use_decoder:
        decoder = ThresholdDecoder(pr.threshold_config, pr.threshold_center)
        predictions = np.array([[decoder.decode(x[0])] for x in raw_outputs])
    else:
        predictions = raw_outputs
    
    stats = Stats(predictions, targets, filenames)

    print('Data:', data)

    if not args.no_filenames:
        fp_files = stats.calc_filenames(False, True, args.threshold)
        fn_files = stats.calc_filenames(False, False, args.threshold)
        print('=== False Positives ===')
        print('\n'.join(fp_files))
        print()
        print('=== False Negatives ===')
        print('\n'.join(fn_files))
        print()
    print(stats.counts_str(args.threshold))
    print()
    print(stats.summary_str(args.threshold))


if __name__ == '__main__':
    main()
