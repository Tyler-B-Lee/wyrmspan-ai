import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def load_model(model_name):
    try:
        print(f"Loading model: {model_name}...")
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_cave_card_string(cave_card: dict) -> str:
    card_text: str = cave_card['text']
    # remove bracket characters
    card_text = card_text.replace("][", " and ")
    card_text = card_text.replace("[", "").replace("]", "")
    return f"Type: CaveCard | WhenPlayed Ability: {card_text}"

def get_dragon_card_string(dragon_card: dict) -> str:
    card_text: str = dragon_card['ability_text']
    # remove bracket characters
    card_text = card_text.replace("][", " and ")
    card_text = card_text.replace("[", "").replace("]", "")
    
    # find ability type string
    ability_type: str = ""
    if 'if_activated' in dragon_card:
        ability_type = "Adventurer"
    elif 'when_played' in dragon_card:
        ability_type = "WhenPlayed"
    elif 'once_per_round' in dragon_card:
        ability_type = "OncePerRound"
    elif 'end_game' in dragon_card:
        if len(ability_type) == 0:
            ability_type = "EndGame"
        else:
            ability_type += ", EndGame"
    
    return f"Type: DragonCard | {ability_type} Ability: {card_text}"

def get_dragon_numerical_tensor(dragon_card: dict) -> torch.Tensor:
    # create zeros tensor
    output = torch.zeros(19, dtype=torch.float32)
    # fill in tensor
    output[0] = dragon_card['VP'] / 10
    output[1] = dragon_card['capacity'] / 5
    output[2] = dragon_card['meat_cost'] / 4
    output[3] = dragon_card['gold_cost'] / 4
    output[4] = dragon_card['crystal_cost'] / 4
    output[5] = dragon_card['milk_cost'] / 4
    output[6] = dragon_card['coin_cost'] # 0 or 1
    output[7] = dragon_card['egg_cost'] # 0 or 1
    # size one-hot encoding
    size_mapping = {'Hatchling': 0, 'Small': 1, 'Medium': 2, 'Large': 3}
    if dragon_card['size'] in size_mapping:
        output[8 + size_mapping[dragon_card['size']]] = 1.0
    # personality one-hot encoding
    personality_mapping = {'Aggressive': 0, 'Helpful': 1, 'Shy': 2, 'Playful': 3}
    if dragon_card['personality'] in personality_mapping:
        output[12 + personality_mapping[dragon_card['personality']]] = 1.0
    # cave preferences
    if dragon_card['crimson_cavern']:
        output[16] = 1.0
    if dragon_card['golden_grotto']:
        output[17] = 1.0
    if dragon_card['amethyst_abyss']:
        output[18] = 1.0
    return output

if __name__ == "__main__":
    model_name = "all-MiniLM-L6-v2"
    model = load_model(model_name)
    if model:
        print("Model loaded successfully")
    else:
        print("Failed to load model")

    # create and save cave card embeddings
    with open('data/cave_cards.json', 'r') as f:
        cave_cards = json.load(f)
    
    strings= []
    for card in cave_cards:
        if 'text' in card:
            card_string = get_cave_card_string(card)
            print(card_string)
            strings.append(card_string)
    embeddings_tensor = model.encode(strings, convert_to_tensor=True)
    # caves do not have numerical vectors, so just use zeros
    vector_part = torch.zeros((embeddings_tensor.shape[0], 19), dtype=torch.float32)
    embeddings_tensor = torch.cat((embeddings_tensor, vector_part), dim=1)
    
    print(f"Final cave embeddings shape: {embeddings_tensor.shape}")
    torch.save(embeddings_tensor, 'data/cave_card_embeddings.pth')

    print(f"Similarities in first few embeddings: {model.similarity(embeddings_tensor[37:42], embeddings_tensor[37:42])}")

    # dragon card embeddings
    with open('data/dragon_cards.json', 'r') as f:
        dragon_cards = json.load(f)
    
    strings= []
    vectors = []
    for card in dragon_cards:
        if 'ability_text' in card:
            card_string = get_dragon_card_string(card)
            print(card_string)
            strings.append(card_string)
            card_vector = get_dragon_numerical_tensor(card)
            print(f"Numerical vector: {card_vector}")
            vectors.append(card_vector)
    embeddings_tensor = model.encode(strings, convert_to_tensor=True)

    all_vectors = torch.stack(vectors)
    embeddings_tensor = torch.cat((embeddings_tensor, all_vectors), dim=1)
    
    print(f"Final dragon embeddings shape: {embeddings_tensor.shape}")
    torch.save(embeddings_tensor, 'data/dragon_card_embeddings.pth')

    print(f"Similarities in first few embeddings: {model.similarity(embeddings_tensor[37:42], embeddings_tensor[37:42])}")