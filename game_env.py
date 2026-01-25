from game_states import *

import torch
import numpy as np


def get_global_info(game_state: SoloGameState) -> dict:
    """
    Get a dictionary of tensors representing the global information of the game environment.
    Each group of related features will be stored in its own tensor within the dictionary.
    These will be translated into input tokens for the model.
    
    The features include:
    - Game Timing
        - Current Round / Game Phase
        - Number of Coins (Turns left)
        - Automa Deck Size
        - Automa Passed This Round
        - Player Score
        - Automa Score
    - Guild Board Status
        - Which guild is active (one-hot)
        - Player guild marker position
        - Automa guild marker position
        - Player guild markers remaining
    - Deck Status
        - Dragon deck size
        - Cave deck size
        - Dragon deck tensor
        - Cave deck tensor
    - Player Resources
        - Dragon Cards in hand
        - Cave Cards in hand
        - Coins
        - Each resource type (meat, gold, crystal, milk)
        - Egg totals by location
    - Automa Status
        - Automa score
        - Automa difficulty
        - Automa dragons in hand
        - Automa caves in hand
        - Automa passed this round
        - Automa decision deck cards (one-hot)
    """
    info_dict = {}

    # Game Timing (17 values)
    timing_tensor = torch.zeros(17, dtype=torch.float32)
    timing_tensor[game_state.board["round_tracker"]["round"]] = 1.0  # One-hot encode current round (0-3)
    current_phase_index = PHASE_INDEX.get(game_state.phase, 0)
    timing_tensor[4 + current_phase_index] = 1.0  # One-hot encode current phase (0-7)
    timing_tensor[12] = game_state.player.coins / 10.0  # Normalize coins
    timing_tensor[13] = len(game_state.automa.decision_deck) / 8.0  # Normalize automa deck (0-8 cards)
    timing_tensor[14] = 1.0 if game_state.automa.passed_this_round else 0.0  # Automa passed this round
    timing_tensor[15] = (game_state.player.score) / 100.0  # Player score (normalized estimate)
    timing_tensor[16] = (game_state.automa.score) / 100.0  # Automa score (normalized estimate)
    info_dict["timing"] = timing_tensor

    # Guild Board Status (51 combined values)
    guild_tensor = torch.zeros(28, dtype=torch.float32)
    guild_index = game_state.board["guild"]["guild_index"]
    # One-hot encode guild (4 possible guilds)
    if guild_index < 4:
        guild_tensor[guild_index] = 1.0
    guild_tensor[4 + game_state.board["guild"]["player_position"]] = 1.0  # Player position on guild track (0-11)
    guild_tensor[16 + game_state.board["guild"]["automa_position"]] = 1.0  # Automa position on guild track
    guild_tensor[27] = game_state.player.guild_markers / 4.0  # Player guild markers remaining
    # Guild ability uses
    guild_ability_tensor = torch.zeros((5, 3), dtype=torch.float32)
    ability_uses = game_state.board["guild"]["ability_uses"]
    for i in range(1, 6):
        uses = ability_uses[i]
        if len(uses) > 0:
            num_uses = min(len(uses), 3)
            guild_ability_tensor[i - 1, num_uses - 1] = 1.0
    # append guild ability uses to the end of guild tensor
    guild_tensor = torch.cat((guild_tensor, guild_ability_tensor.flatten()), dim=0)
    # extra info of end game point ability
    end_game_ability_tensor = torch.zeros((2,4), dtype=torch.float32)
    for player in range(2):
        count = ability_uses[5][2:].count(player)
        if count > 0:
            end_game_ability_tensor[player, count - 1] = 1.0
    guild_tensor = torch.cat((guild_tensor, end_game_ability_tensor.flatten()), dim=0)
    info_dict["guild_status"] = guild_tensor

    # Deck Status (2 + 183 + 75 = 260 values)
    deck_tensor = torch.zeros(2, dtype=torch.float32)
    deck_tensor[0] = len(game_state.dragon_deck) / 183.0  # Dragon deck size
    deck_tensor[1] = len(game_state.cave_deck) / 75.0  # Cave deck size
    # combine dragon and cave deck tensors from game state
    deck_tensor = torch.cat((
        deck_tensor,
        game_state.dragon_deck_tensor,
        game_state.cave_deck_tensor
    ), dim=0)
    info_dict["deck_status"] = deck_tensor

    # Player Resources (13 values: hand sizes, coins, resources, eggs)
    player_tensor = torch.zeros(13, dtype=torch.float32)
    player_tensor[0] = len(game_state.player.dragon_hand) / 10.0  # Dragon cards in hand
    player_tensor[1] = len(game_state.player.cave_hand) / 10.0  # Cave cards in hand
    player_tensor[2] = game_state.player.coins / 10.0  # Coins
    # Resources (meat, gold, crystal, milk)
    for i, resource in enumerate(RESOURCES):
        player_tensor[3 + i] = game_state.player.resources[resource] / 10.0  # Normalize resource counts
    # Egg totals by location
    player_tensor[7] = game_state.player.egg_totals["mat_slots"] / 2.0
    player_tensor[8] = game_state.player.egg_totals["crimson_cavern"] / 12.0
    player_tensor[9] = game_state.player.egg_totals["golden_grotto"] / 12.0
    player_tensor[10] = game_state.player.egg_totals["amethyst_abyss"] / 12.0
    player_tensor[11] = sum(game_state.player.egg_totals.values()) / 40.0  # Total eggs
    player_tensor[12] = sum(game_state.player.num_dragons_played.values()) / 12.0  # Dragons on mat
    info_dict["player_resources"] = player_tensor

    # Automa Status (25 values)
    automa_tensor = torch.zeros(25, dtype=torch.float32)
    automa_tensor[game_state.automa.difficulty] = 1.0  # Automa difficulty level
    automa_tensor[6] = game_state.automa.score / 100.0  # Automa score
    automa_tensor[7] = len(game_state.automa.dragons) / 16.0  # Automa dragons collected
    automa_tensor[8] = len(game_state.automa.caves) / 10.0  # Automa caves collected
    automa_tensor[9] = 1.0 if game_state.automa.passed_this_round else 0.0  # Automa passed this round
    for card_i in game_state.automa.decision_deck:
        automa_tensor[10 + card_i] = 1.0  # One-hot encode automa decision deck cards
    info_dict["automa_status"] = automa_tensor

    return info_dict

def get_player_board_info(game_state: SoloGameState) -> dict:
    """
    Get a dictionary of tensors representing the player's board information.
    Each group of related features will be stored in its own tensor within the dictionary.
    These will be translated into input tokens for the model.
    
    For each of the 12 mat slots, we will have 3 tensors containing:
    - 1st: Slot Type
        - Is slot empty, excavated, or has a dragon
    - 2nd: Dragon on Slot
        - Which dragon is on the slot, if any
    - 3rd: Slot Details
        - Number of eggs on slot
        - Number of dragons played of that type
        - Is this a hatchling
        - Is this hatchling grown
        - What resources are cached on this slot
        - Number of tucked dragons under this slot
    """