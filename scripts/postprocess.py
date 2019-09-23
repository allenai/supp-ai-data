"""
Compile BERT-DDI results
Create final dicts for site

"""

import os
import sys
import json
import multiprocessing


LOG_FILE = 'config/log.json'
NUM_PROCESSES = multiprocessing.cpu_count() // 2

if __name__ == '__main__':
    # read preprocessing log file
    with open(LOG_FILE, 'r') as f:
        log_dict = json.load(f)

    ddi_output_dir = log_dict['ddi_output_dir']
    output_file = log_dict['output_file']

    # TODO: compile BERT-DDI results

    # TODO: form final dicts

    # TODO: tar and zip output files