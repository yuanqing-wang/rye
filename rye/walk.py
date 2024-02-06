import torch

def next_node(current_node, probability):
    next_probability = probability[current_node]
    return torch.multinomial(next_probability, 1).squeeze(-1)

def generate_walk(probability, length):
    walk = []
    current_node = torch.arange(probability.shape[0])
    for _ in range(length):
        print(current_node)
        current_node = next_node(current_node, probability)
        walk.append(current_node)
    walk = torch.stack(walk, dim=-2)
    return walk


 