"""
This file converts a safetensors file containing Hanabi game data
into a human-readable JSON file.

Each tensor is loaded, converted to Python lists, and saved into a
.json file next to the input dataset.
"""
import json
import os
from safetensors import safe_open
from typing import Dict, Any

def print_tensor_shapes(input_path: str) -> None:
    """Prints the shapes of the tensors in the safetensors file."""
    with safe_open(input_path, framework="pt") as f:
        print("=== Tensor Shapes ===")
        for name in f.keys():
            tensor = f.get_tensor(name)
            print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print("=====================\n")

def extract_game_data(input_path: str, output_path: str, game_idx: int) -> None:
    """Extracts a specific game and saves it as a JSON file."""
    game_data: Dict[str, Any] = {}
    with safe_open(input_path, framework="pt") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            if tensor.dim() == 0:
                game_data[name] = tensor.item()
            else:
                entry = tensor[game_idx]
                if entry.dim() == 0:
                    game_data[name] = entry.item()
                else:
                    game_data[name] = entry.tolist()

    with open(output_path, "w") as fp:
        json.dump(game_data, fp, indent=2)
    
    print(f"Extracted game {game_idx + 1} saved to {output_path}")

def main() -> None:
    """Main function to run the game data extraction."""
    INPUT_PATH = "data/3_player_games_val.safetensors"
    OUTPUT_DIR = "readable_data"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print_tensor_shapes(INPUT_PATH)

    with safe_open(INPUT_PATH, framework="pt") as f:
        num_games = f.get_tensor("actions").shape[0]

    for i in range(num_games):
        output_path = os.path.join(OUTPUT_DIR, f"game_{i + 1}.json")
        extract_game_data(INPUT_PATH, output_path, i)

if __name__ == "__main__":
    main()