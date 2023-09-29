#!/bin/bash

# run this script in your *activated* virtual environment

for i in {1..8}; do
  'python3' set_config.py "ea1-fps" "ea2-closest-to-mean" "$i" 100 100 1 -1
  'python3' compare_specialist.py
  'python3' plotting.py
done