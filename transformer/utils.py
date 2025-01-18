import torch
import torch.nn.functional as F

def apply_mask_to_tensor(x, mask):
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (torch.Tensor): Tensor of shape (bs, c, h, w)
        mask (torch.Tensor): Tensor of shape (bs, num_tokens)

    Returns:
        torch.Tensor: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
    bs, c, h, w = x.shape

    mask = mask.view(bs, h, w)

    # Expand the mask to cover each patch
    # (bs, h, w) -> (bs, 1, h, w)
    mask = mask.unsqueeze(1)  # Add channel dimension

    # Apply the mask to the input tensor
    x = x * mask

    return x

def unpatchify(x, patch_size, height, width):
    """
    Reconstructs images from patches.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_tokens, patch_size * patch_size * in_channels)
        patch_size (int): Size of each patch.
        height (int): Original image height.
        width (int): Original image width.

    Returns:
        torch.Tensor: Reconstructed image of shape (bs, in_channels, height, width)
    """
    bs, num_tokens, patch_dim = x.shape
    H, W = patch_size
    in_channels = patch_dim // (H * W)

    # Calculate the number of patches along each dimension
    num_tokens_h = height // H
    num_tokens_w = width // W

    # Ensure num_tokens equals num_tokens_h * num_tokens_w
    assert num_tokens == num_tokens_h * num_tokens_w, "Mismatch in number of patches."

    # Reshape x to (bs, num_tokens_h, num_tokens_w, H, W, in_channels)
    x = x.view(bs, num_tokens_h, num_tokens_w, H, W, in_channels)

    # Permute x to (bs, num_tokens_h, H, num_tokens_w, W, in_channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape x to (bs, height, width, in_channels)
    reconstructed = x.view(bs, height, width, in_channels)

    # Permute back to (bs, in_channels, height, width)
    reconstructed = reconstructed.permute(0, 3, 1, 2).contiguous()

    return reconstructed

def random_mask(bs: int, height: int, width: int, mask_ratio: float) -> torch.Tensor:
    """
    Generates a random mask for image tokens. Randomly selects tokens to mask.

    Args:
        bs (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        mask_ratio (float): Ratio of tokens to mask. Ranges from 0 to 1. mask_ratio * 100 = percentage of 1s in the mask

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_tokens) with values in {0, 1}.
    """
    num_tokens = height * width
    num_tokens_to_mask = int(num_tokens * mask_ratio)
    
    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_tokens)
    
    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)
    
    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_tokens)
    
    # Set the first num_tokens_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_tokens_to_mask]] = 0
    
    # Ensure the final shape is (bs, num_tokens)
    mask = mask.view(bs, num_tokens)

    return mask

def remove_masked_tokens(tokens, mask):
    """
    Removes the masked tokens from the tokens tensor while preserving batch dimensions.
    Returned tensor will have shape (bs, number_of_unmasked_tokens, embed_dim).
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()
    mask = mask.logical_not()

    # Get batch size and embed dimension
    bs, num_tokens, embed_dim = tokens.shape

    # Expand mask to match the shape of tokens for correct indexing
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    # Use masked_select and reshape to maintain batch size
    unmasked_tokens = torch.masked_select(tokens, ~mask).view(bs, -1, embed_dim)

    return unmasked_tokens

def add_masked_tokens(tokens, mask):
    """
    Adds the masked tokens to the tokens tensor.
    Returned tensor will have shape (bs, num_tokens, embed_dim).
    The missing tokens will be filled with 0s.
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Get the total number of tokens and embed dimension
    bs, num_tokens, embed_dim = mask.shape[0], mask.shape[1], tokens.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the tokens tensor
    full_tokens = torch.zeros(bs, num_tokens, embed_dim, device=tokens.device, dtype=tokens.dtype)

    # Iterate over each batch and place unmasked tokens back in their original positions
    for i in range(bs):
        # Use the mask to place unmasked tokens back in the correct positions
        full_tokens[i, mask[i]] = tokens[i].to(full_tokens.dtype)

    return full_tokens