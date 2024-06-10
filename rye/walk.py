import torch

def next_node(current_node, probability):
    batch_shape = probability.shape[:-2]
    probability = probability.reshape(-1, *probability.shape[-2:])
    current_node = current_node.reshape(-1, current_node.shape[-1])
    next_probability = probability[torch.arange(probability.shape[-3])[:, None], current_node]
    num_nodes = next_probability.shape[-1]
    next_probability = next_probability.view(-1, num_nodes)
    next_node = torch.multinomial(next_probability, 1)
    next_node = next_node.view(*batch_shape, -1).squeeze(-1)
    return next_node

def generate_walk(probability, length):
    # (BATCH_SIZE, NUM_NODES, NUM_NODES)
    current_node = torch.arange(probability.shape[-1], device=probability.device)
    current_node = current_node.expand(probability.shape[:-2] + current_node.shape)
    walk = [current_node]
    for _ in range(length):
        current_node = next_node(current_node, probability)
        walk.append(current_node)
    walk = torch.stack(walk, dim=-2)
    return walk


 