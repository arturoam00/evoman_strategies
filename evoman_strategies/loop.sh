#!/bin/bash

# run this script in your *activated* virtual environment

for i in {1..8}; do
  'python3' set_config.py "EA1" "EA2" $i 100 30 1 -1
  'python3' compare_specialist.py
  'python3' plotting.py
done