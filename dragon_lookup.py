import pprint

from game_states import DRAGON_CARDS

def lookup_dragon_card(card_id):
    """
    Looks up a dragon card by its index and pretty-prints the card details.
    """
    card = DRAGON_CARDS[card_id]
    pprint.pprint(card)

if __name__ == "__main__":
    while True:
        try:
            card_id = int(input("Enter the dragon card ID (1 to 183) or 0 to exit: "))
            if card_id == 0:
                print("Exiting...")
                break
            lookup_dragon_card(card_id)
        except ValueError:
            print("Please enter a valid integer.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")