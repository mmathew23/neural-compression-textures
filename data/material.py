import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize


keyword_order = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]
# Map of keywords to texture types
texture_keywords = {
    keyword_order[0]: ["diffuse", "albedo", "color", "diff"],
    keyword_order[1]: ["normal", "nor_gl"],
    keyword_order[2]: ["roughness", "rough"],
    keyword_order[3]: ["occlusion", "ao", "ambient"],
    keyword_order[4]: ["metallic", "metalness"],
    keyword_order[5]: ["specular"],
    keyword_order[6]: ["displacement", "disp"],
    # add other texture types and their possible keywords if needed
}

texture_configs = {
    keyword_order[0]: {"expected_channels": 3, "mode": "RGB"},
    keyword_order[1]: {"expected_channels": 3, "mode": "RGB"},
    keyword_order[2]: {"expected_channels": 1, "mode": "L"},
    keyword_order[3]: {"expected_channels": 1, "mode": "L"},
    keyword_order[4]: {"expected_channels": 1, "mode": "L"},
    keyword_order[5]: {"expected_channels": 1, "mode": "L"},
    keyword_order[6]: {"expected_channels": 1, "mode": "L"},
    # add other texture types if needed
}


def identify_texture_type(filename):
    filename_lower = filename.lower()

    # Identify the texture type based on keywords in the filename
    for texture_type, keywords in texture_keywords.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return texture_type

    raise ValueError(f"Could not identify texture type from filename: {filename}")


def process_images(directory, resolution, dtype=torch.float32):
    textures = {}
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # add other file types if needed
            file_path = os.path.join(directory, filename)

            with Image.open(file_path) as img:
                texture_type = identify_texture_type(filename)
                if texture_type not in texture_configs:
                    raise ValueError(f"Unknown texture type: {texture_type}")

                img = img.convert(texture_configs[texture_type]['mode'])
                img = img.resize((resolution, resolution), Image.BICUBIC)

                tensor = to_tensor(img).to(dtype)

                if tensor.shape[0] != texture_configs[texture_type]['expected_channels']:
                    if tensor.shape[0] > texture_configs[texture_type]['expected_channels']:
                        print(f"Unexpected number of channels for {texture_type}. Expected {texture_configs[texture_type]['expected_channels']}, got {tensor.shape[0]}")
                        print(f"Defaulting to {texture_configs[texture_type]['expected_channels']} channels")
                        tensor = tensor[:texture_configs[texture_type]['expected_channels']]
                    else:
                        # Usually when expected channels doesnt match the actual, it's because there is an alpha channel
                        # or it's grayscale but opening as rgb. In these cases the expected < than actual and we can just
                        # slice off the extra channeles, but otherwise we don't know how to handle it
                        raise ValueError(f"Unexpected number of channels for {texture_type}. Expected {texture_configs[texture_type]['expected_channels']}, got {tensor.shape[0]}")

                textures[texture_type] = tensor

    texture_list = []

    for texture_type in keyword_order:
        if texture_type in textures and textures[texture_type] is not None:
            texture_list.append(textures[texture_type])

    tensor = torch.cat(texture_list, dim=0)
    c = tensor.shape[0]
    normalize_constant = torch.tensor([0.5 for i in range(c)], dtype=dtype)
    return normalize(tensor, normalize_constant, normalize_constant)