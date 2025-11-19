#!/bin/bash

# Oryx har allerede satt opp Python-miljøet og installert requirements.
# Du trenger bare å starte serveren.
python -m hypercorn editor_api:app --bind 0.0.0.0:${PORT:-8000}