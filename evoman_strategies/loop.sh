#!/bin/bash

# run this script in your *activated* virtual environment

for i in {2,6,7}; do
  'python3' set_config.py "FPS" "MCS" "$i" 100 40 1 -1
  'python3' compare_specialist.py
  'python3' plotting.py
done
