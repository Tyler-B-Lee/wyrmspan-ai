import time
import random
import sys

# comparison of methods for performance

# Storing cards in a set vs list
deck = set(range(1, 184))  # Set of dragon cards IDs
deck_list = list(deck)

# check memory usage
import sys
print(f"Size of set: {sys.getsizeof(deck)} bytes")
print(f"Size of list: {sys.getsizeof(deck_list)} bytes")
# check performance of drawing cards from set vs list
print("Picking random cards from set vs list")
avg_time_set = 0
avg_time_list = 0
num_iterations = 10000
for _ in range(num_iterations):
    start_time = time.time()
    random.sample(deck, 5)
    end_time = time.time()
    avg_time_set += (end_time - start_time)
avg_time_set /= num_iterations
print(f"Average time for set: {avg_time_set} seconds")

for _ in range(num_iterations):
    start_time = time.time()
    random.sample(deck_list, 5)
    end_time = time.time()
    avg_time_list += (end_time - start_time)
avg_time_list /= num_iterations
print(f"Average time for list: {avg_time_list} seconds")

# removing cards from set vs list
# check performance of removing cards from set vs list
print("\nRemoving random cards from set vs list")
avg_time_set = 0
avg_time_list = 0
num_iterations = 10000
for _ in range(num_iterations):
    deck = set(range(1, 184))  # Set of dragon cards IDs
    start_time = time.time()
    deck -= set(random.sample(deck, 5))
    end_time = time.time()
    avg_time_set += (end_time - start_time)
avg_time_set /= num_iterations
print(f"Average time for set: {avg_time_set} seconds")

for _ in range(num_iterations):
    deck_list = list(range(1, 184))  # Set of dragon cards IDs
    start_time = time.time()
    for card in random.sample(deck_list, 5):
        deck_list.remove(card)
    end_time = time.time()
    avg_time_list += (end_time - start_time)
avg_time_list /= num_iterations
print(f"Average time for list: {avg_time_list} seconds")

# making copies of set vs list
# check performance of making copies of set vs list
print("\nMaking copies of set vs list")
avg_time_set = 0
avg_time_list = 0
num_iterations = 10000
for _ in range(num_iterations):
    deck = set(range(1, 184))  # Set of dragon cards IDs
    start_time = time.time()
    deck_copy = deck.copy()
    end_time = time.time()
    avg_time_set += (end_time - start_time)
avg_time_set /= num_iterations
print(f"Average time for set: {avg_time_set} seconds")

for _ in range(num_iterations):
    deck_list = list(range(1, 184))  # Set of dragon cards IDs
    start_time = time.time()
    deck_list_copy = deck_list.copy()
    end_time = time.time()
    avg_time_list += (end_time - start_time)
avg_time_list /= num_iterations
print(f"Average time for list: {avg_time_list} seconds")