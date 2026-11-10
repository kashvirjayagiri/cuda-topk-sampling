import torch

#DONE BY JOEL SEDILLO
def sample_reference(scores, token_ids, temperature, random_values):
    """
    scores:        [B, k] selected scores
    token_ids:     [B, k] original token IDs, in matching order
    temperature:   positive number
    random_values: [B] supplied numbers in [0, 1)

    Returns: [B] chosen original token IDs
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    # 1. Apply temperature and convert scores to probabilities.python -m pip install numpy
    probabilities = torch.softmax(scores / temperature, dim=-1)

    # 2. Calculate the running totals for each row.
    cumulative = torch.cumsum(probabilities, dim=-1)

    # 3. Handle rounding so the final boundary is exactly 1.
    cumulative = cumulative.clamp(max=1.0)
    cumulative[:, -1] = 1.0

    # 4. Find the first cumulative probability greater than
    #    each row's supplied random value.
    positions = torch.searchsorted(
        cumulative,
        random_values.unsqueeze(-1),
        right=True,
    )

    # 5. Return original token IDs rather than candidate positions.
    sampled_ids = token_ids.gather(dim=1, index=positions)

    return sampled_ids.squeeze(1)


if __name__ == "__main__":
    scores = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32)
    token_ids = torch.tensor([[17, 42, 90]], dtype=torch.int32)
    random_values = torch.tensor([0.8], dtype=torch.float32)

    result = sample_reference(
        scores,
        token_ids,
        temperature=1.0,
        random_values=random_values,
    )

    print("Selected token IDs:", result.tolist())
    assert result.tolist() == [42]