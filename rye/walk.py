import torch

def next_node(current_node, probability):
    next_probability = probability[current_node]
    repeat = next_probability.shape[-3]
    num_nodes = next_probability.shape[-2]
    next_probability = next_probability.view(-1, num_nodes)
    next_node = torch.multinomial(next_probability, 1)
    next_node = next_node.view(repeat, num_nodes)
    return next_node

def generate_walk(probability, length, repeat=1):
    current_node = torch.arange(probability.shape[-2], device=probability.device).unsqueeze(0).expand(repeat, -1)
    walk = [current_node]
    for _ in range(length):
        current_node = next_node(current_node, probability)
        walk.append(current_node)
    walk = torch.stack(walk, dim=-2)
    return walk


 