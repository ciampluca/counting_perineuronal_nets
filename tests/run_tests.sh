#!/bin/bash

# call from repo root
python -m unittest -v tests/test_data_loaders.py
python -m unittest -v tests/test_models.py