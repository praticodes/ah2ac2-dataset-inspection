"""
This script traces a single game of Hanabi from a safetensors file.

It loads the game data, simulates each action step-by-step, and prints a human-readable trace of the game to the console.
This is useful for understanding the game flow and debugging game-playing agents.
"""
import json
import os
from safetensors import safe_open
import torch
from typing import List, Dict, Tuple, Any, TextIO

def load_constants(path_to_logging_constants: str = "logging_constants.json") -> Dict[str, Any]:
    """
    Loads constants from a JSON file for action-description and num-color mappings. 
    This is used to translate action values to action descriptions and color values to color names when logging games.

    Args:
        path_to_logging_constants (str): The path to the JSON file.

    Returns:
        dict: A dictionary of constants.
    """
    # load the JSON file from the specified path
    with open(path_to_logging_constants, "r") as f:
        constants = json.load(f)

    # Convert color map from string to int
    constants["COLOR_MAP"] = {int(k): v for k, v in constants["COLOR_MAP"].items()}

    return constants


class InvalidGameError(Exception):
    """Custom exception for invalid game data."""
    pass


def load_game_data(path_to_game: str, game_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Loads the game at the specified index from the safetensors file.

    Args:
        path_to_game (str): The path to the safetensors file.
        game_idx (int): The index of the game to load.

    Returns:
        tuple: A tuple containing the actions, deck, number of actions,
               and number of players for the specified game.
    """
    with safe_open(path_to_game, framework="pt") as f:
        num_games = f.get_tensor("actions").shape[0]
        if game_idx >= num_games:
            raise InvalidGameError(f"Game index {game_idx} is out of bounds. The dataset contains {num_games} games.")
        actions = f.get_tensor("actions")[game_idx]
        deck = f.get_tensor("decks")[game_idx]
        num_actions = int(f.get_tensor("num_actions")[game_idx].item())
        num_players = int(f.get_tensor("num_players").item())
        score = int(f.get_tensor("scores")[game_idx].item())

    return actions, deck, num_actions, num_players, score


def card_str(card_tuple: Tuple[int, int], color_map: Dict[int, str]) -> str:
    """
    Returns a human-readable string for a card tuple.

    Args:
        card_tuple (tuple): A tuple representing a card (color, rank).
        color_map (dict): A mapping from color index to color name.

    Returns:
        str: A string representation of the card.
    """
    color, rank = card_tuple
    return f"{color_map[color]} {rank + 1}" # Add one to rank to convert from 0- to 1-based indexing


def initialize_game_state(game_deck: torch.Tensor,
                          num_players: int,
                          color_map: Dict[int, str]) -> tuple[list[list[str]], list[Any], dict[str, int], int]:
    """
    Initializes the game state (hands, piles, etc.).

    Args:
        game_deck (torch.Tensor): The deck of cards for the game.
        num_players (int): The number of players in the game.
        color_map (dict): A mapping from color index to color name.

    Returns:
        tuple: A tuple containing the initial hands, discard pile,
               played pile, and deck draw pointer.
    """
    # Assign each player 5 cards from the deck, in order: Player 1 gets the first 5, Player 2 gets the next 5, etc.
    hands = [
        [card_str(tuple(card.tolist()), color_map) for card in game_deck[i * 5: (i + 1) * 5]]
        for i in range(num_players)
    ]

    # Initialize an empty discard pile and played stacks
    discard_pile = []
    played_pile = {color: 0 for color in color_map.values()}

    # Set a pointer to the next card to be drawn from the deck: starts at the next unassigned card.
    deck_draw_ptr = num_players * 5
    return hands, discard_pile, played_pile, deck_draw_ptr


def print_game_state(hands: List[List[str]], discard_pile: List[str], played_pile: List[str], file: TextIO) -> None:
    """
    Prints the current state of the game.

    Args:
        hands (list): A list of lists representing the players' hands.
        discard_pile (list): A list of cards in the discard pile.
        played_pile (list): A list of cards in the played pile.
        file (TextIO): The file to write the output to.
    """
    # Print player hands
    for i, hand in enumerate(hands):
        print(f"      Player {i + 1} hand: {hand}", file=file) # Add one to i to convert from 0- to 1-based indexing

    # Print discard pile
    print(f"      Discard pile: {discard_pile}", file=file)

    # Format and print played pile: show the highest rank per color, but '_' if none played
    played_str = ", ".join(f"{color} {rank if rank > 0 else '_'}"
                           for color, rank in played_pile.items())
    print(f"      Played pile:  {played_str}\n", file=file)


def handle_discard(action_value: int,
                   hands: List[List[str]],
                   acting_player_idx: int,
                   discard_pile: List[str],
                   game_deck: torch.Tensor,
                   deck_draw_ptr: int,
                   color_map: Dict[int, str],
                   file: TextIO) -> int:
    """
    Handles a discard action.

    Args:
        action_value (int): The value of the discard action.
        hands (list): The current hands of the players.
        acting_player_idx (int): The index of the acting player.
        discard_pile (list): The current discard pile.
        game_deck (torch.Tensor): The deck of cards for the game.
        deck_draw_ptr (int): The pointer to the next card to be drawn.
        color_map (dict): A mapping from color index to color name.
        file (TextIO): The file to write the output to.

    Returns:
        int: The updated deck draw pointer.
    """
    # Validate discard card index
    idx = action_value
    if idx >= len(hands[acting_player_idx]):
        raise InvalidGameError("Game data contains an invalid discard index")

    # Add requested card to discard pile
    card = hands[acting_player_idx].pop(idx)
    discard_pile.append(card)
    print(f"   -> Discarded {card}", file=file)

    # Draw a new card from the deck if all 50 cards not yet used
    more_cards = deck_draw_ptr < 50
    if more_cards:
        new_card = card_str(tuple(game_deck[deck_draw_ptr].tolist()), color_map)
        hands[acting_player_idx].append(new_card)
        print(f"   -> Drew {new_card}", file=file)
        deck_draw_ptr += 1

    return deck_draw_ptr

def handle_play(action_value: int,
                hands: List[List[str]],
                acting_player_idx: int,
                played_pile: List[str],
                game_deck: torch.Tensor,
                deck_draw_ptr: int,
                color_map: Dict[int, str],
                file: TextIO) -> int:
    """
    Handles a play action.

    Args:
        action_value (int): The value of the play action.
        hands (list): The current hands of the players.
        acting_player_idx (int): The index of the acting player.
        played_pile (list): The current played pile.
        game_deck (torch.Tensor): The deck of cards for the game.
        deck_draw_ptr (int): The pointer to the next card to be drawn.
        color_map (dict): A mapping from color index to color name.
        file (TextIO): The file to write the output to.

    Returns:
        int: The updated deck draw pointer.
    """
    # Validate play card index
    idx = action_value - 5
    if idx >= len(hands[acting_player_idx]):
        raise InvalidGameError("Game data contains an invalid play index")

    # Remove card from hand to indicate played
    card = hands[acting_player_idx].pop(idx)
    print(f"   -> Played {card}", file=file)

    # Extract card info
    color, rank_str = card.split()
    rank = int(rank_str)

    # Update the played pile
    if rank > played_pile[color]:
        played_pile[color] = rank

    # Draw a new card from the deck if all 50 cards not yet used
    more_cards = deck_draw_ptr < 50
    if more_cards:
        new_card = card_str(tuple(game_deck[deck_draw_ptr].tolist()), color_map)
        hands[acting_player_idx].append(new_card)
        print(f"   -> Drew {new_card}", file=file)
        deck_draw_ptr += 1

    return deck_draw_ptr

def trace_game(game_idx: int,
               game_actions: torch.Tensor,
               game_deck: torch.Tensor,
               num_actions: int,
               num_players: int,
               score: int,
               action_descriptions: List[str],
               color_map: Dict[int, str],
               file: TextIO) -> None:
    """
    Traces a single game of Hanabi, printing each step.

    Args:
        game_idx (int): The index of the game being traced.
        game_actions (torch.Tensor): The actions taken in the game.
        game_deck (torch.Tensor): The deck of cards for the game.
        num_actions (int): The number of actions in the game.
        num_players (int): The number of players in the game.
        action_descriptions (list): A list of strings describing the actions.
        color_map (dict): A mapping from color index to color name.
        file (TextIO): The file to write the output to.
    """
    # Initialize game state
    hands, discard_pile, played_pile, deck_draw_ptr = initialize_game_state(
        game_deck, num_players, color_map
    )

    print(f"=============== HANABI GAME {game_idx + 1} TRACE ===============\n", file=file)

    # Trace game actions one by one; logging each step and its impact on the game state
    for action_num in range(num_actions):
        # Get action vector for the current step
        action_vec = game_actions[action_num]

        # The current player is the one with a non-30 action
        non_30 = (action_vec != 30).nonzero(as_tuple=False)
        if len(non_30) == 0:
            continue

        # Interpret the current player's action
        acting_player_idx = int(non_30.item())
        action_value = int(action_vec[acting_player_idx].item())

        # Validate action value and get action description if valid; else raise exception
        if action_value < 0 or action_value > 30:
            raise InvalidGameError("Game data contains an invalid action value")
        action_text = action_descriptions[action_value]

        print(f"Step {action_num + 1}: Player {acting_player_idx+1} — {action_text}", file=file) # Add one for 1-based indexing

        # Classify action and handle it accordingly
        if 0 <= action_value <= 4:
            # Discard action
            deck_draw_ptr = handle_discard(action_value, hands, acting_player_idx, discard_pile, game_deck, deck_draw_ptr, color_map, file)
        elif 5 <= action_value <= 9:
            # Play action
            deck_draw_ptr = handle_play(action_value, hands, acting_player_idx, played_pile, game_deck, deck_draw_ptr, color_map, file)
        elif 10 <= action_value <= 29:
            # Hint action (no change to hands)
            print("   -> Hint action (no change to hands)", file=file)

        print_game_state(hands, discard_pile, played_pile, file)

    print(f"Final Score: {score}", file=file)
    print("=============== END OF GAME TRACE ===============", file=file)

def main() -> None:
    """Main function to run the game trace."""
    # Get all mappings
    CONSTANTS_MAP = load_constants()
    ACTION_DESCRIPTIONS = CONSTANTS_MAP["ACTION_DESCRIPTIONS"]
    COLOR_MAP = CONSTANTS_MAP["COLOR_MAP"]

    # Specify the path to the Hanabi games
    PATH_TO_GAMES = "data/3_player_games_val.safetensors"

    # Create a logs directory (if not already created)
    LOGS_DIR = "logs"
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    # Read the total number of games
    with safe_open(PATH_TO_GAMES, framework="pt") as f:
        num_games = f.get_tensor("actions").shape[0]

    for i in range(num_games):
        log_path = os.path.join(LOGS_DIR, f"game_{i + 1}.txt")
        with open(log_path, "w") as log_file:
            try:
                actions, deck, num_actions, num_players, score = load_game_data(PATH_TO_GAMES, i)
                trace_game(i, actions, deck, num_actions, num_players, score, ACTION_DESCRIPTIONS, COLOR_MAP, log_file)
            except InvalidGameError as e:
                print(f"Error tracing game {i}: {e}", file=log_file)

if __name__ == "__main__":
    main()
