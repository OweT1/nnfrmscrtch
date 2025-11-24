#!/bin/bash
pip install -r requirements.txt

kaggle competitions download -c digit-recognizer

python -m scripts.data