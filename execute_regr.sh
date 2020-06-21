#!/usr/bin/env bash


EPOCHS=1000

python3 Regression.py --time-frame daily --epochs $EPOCHS -p -s

python3 Regression.py --time-frame weekly --epochs $EPOCHS -p -s

python3 Regression.py --time-frame monthly --epochs $EPOCHS -p -s

python3 Regression.py --time-frame seasonally --epochs $EPOCHS -p -s




