import torch

def next_node(current_node, probability):
    # next_probability = probability[current_node]
    next_probability = torch.gather(probability, -1, current_node)



    num_nodes = next_probability.shape[-1]
    batch_shape = next_probability.shape[:-1]
    next_probability = next_probability.view(-1, num_nodes)
    next_node = torch.multinomial(next_probability, 1)
    next_node = next_node.view(*batch_shape, -1)
    return next_node

def generate_walk(probability, length, repeat=1):
    # (BATCH_SIZE, NUM_NODES, NUM_NODES)
    import pdb; pdb.set_trace()
    current_node = torch.arange(probability.shape[-2], device=probability.device).unsqueeze(0).expand(repeat, -1)
    current_node = current_node.broadcast_to((*probability.shape[:-2], *current_node.shape))
    walk = [current_node]
    for _ in range(length):
        current_node = next_node(current_node, probability)
        walk.append(current_node)
    walk = torch.stack(walk, dim=-2)
    return walk


 