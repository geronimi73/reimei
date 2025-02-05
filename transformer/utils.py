import torch
import torch.nn.functional as F

def apply_mask_to_tensor(x, mask, patch_size):
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (torch.Tensor): Tensor of shape (bs, c, h, w)
        mask (torch.Tensor): Tensor of shape (bs, num_patches)
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
    bs, c, h, w = x.shape
    num_patches_h = h // patch_size[0]
    num_patches_w = w // patch_size[1]

    # Ensure that height and width are divisible by patch_size
    assert h % patch_size[0] == 0 and w % patch_size[1] == 0, "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(h, w, patch_size)

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.view(bs, num_patches_h, num_patches_w)

    # Expand the mask to cover each patch
    # (bs, num_patches_h, num_patches_w) -> (bs, 1, h, w)
    mask = mask.unsqueeze(1)  # Add channel dimension
    mask = mask.repeat(1, 1, patch_size[0], patch_size[1])  # Repeat for patch_size
    mask = mask.view(bs, 1, h, w)  # Reshape to (bs, 1, h, w)

    # Apply the mask to the input tensor
    x = x * mask

    return x

def random_mask(bs: int, height: int, width: int, patch_size: tuple[int, int], mask_ratio: float) -> torch.Tensor:
    """
    Generates a random mask for image tokens. Randomly selects tokens to mask.

    Args:
        bs (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        patch_size (tuple[int, int]): Patch size.
        mask_ratio (float): Ratio of tokens to mask. Ranges from 0 to 1. mask_ratio * 100 = percentage of 1s in the mask

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_tokens) with values in {0, 1}.
    """
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)
    
    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_patches)
    
    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)
    
    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_patches)
    
    # Set the first num_patches_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_patches_to_mask]] = 0
    
    # Ensure the final shape is (bs, num_patches)
    mask = mask.view(bs, num_patches)

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