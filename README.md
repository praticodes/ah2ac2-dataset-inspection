# Hanabi Game Inspector

This repository contains Python scripts to inspect and trace Hanabi game data from the [AH2AC2 Hugging Face Datasets](https://huggingface.co/datasets/ah2ac2/datasets/).

## Files

- **`inspect_game.py`**: This script extracts game data from a `.safetensors` file and saves it into a human-readable JSON format. It iterates through all the games in the dataset and saves each game as a separate JSON file in the `readable_data` directory.
- **`trace_game.py`**: This script provides a detailed trace of each game in the dataset. It simulates each game step-by-step, showing the actions taken by each player and the state of the game at each turn. The traces are saved as text files in the `logs` directory.
- **`constants.json`**: This file contains the mappings for action descriptions and colors used in the game.
- **`requirements.txt`**: This file lists the Python dependencies required to run the scripts in this repository.

## How to Use

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the scripts:**
    *   To extract the game data into JSON files, run:
        ```bash
        python3 inspect_game.py
        ```
        This will create a `readable_data` directory with a JSON file for each game. Ensure the specified safetensors file is correct before running this script.

    *   To generate a trace of each game, run:
        ```bash
        python3 trace_game.py
        ```
        This will create a `logs` directory with a text file for each game trace. Ensure the specified safetensors file is correct before running this script.

## Attributions
The game data inspected is from [AH2AC2 Hugging Face Datasets](https://huggingface.co/datasets/ah2ac2/datasets/). All credit for the data goes to the authors of the dataset and the paper cited below.

Dizdarević, T., Hammond, R., Gessler, T., Calinescu, A., Cook, J., Gallici, M., Lupu, A., Muglich, D., Forkel, J., & Foerster, J. (2025). Ad-Hoc Human-AI Coordination Challenge. arXiv preprint arXiv:2506.21490.