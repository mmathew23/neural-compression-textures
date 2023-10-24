import os
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.utils import make_grid


class Material:
    def __init__(self):
        self.keyword_order = ["diffuse", "normal", "roughness", "occlusion", "metallic", "specular", "displacement"]
        self.texture_keywords = {
            self.keyword_order[0]: ["diffuse", "albedo", "color", "diff"],
            self.keyword_order[1]: ["normal", "nor_gl"],
            self.keyword_order[2]: ["roughness", "rough"],
            self.keyword_order[3]: ["occlusion", "ao", "ambient"],
            self.keyword_order[4]: ["metallic", "metalness"],
            self.keyword_order[5]: ["specular"],
            self.keyword_order[6]: ["displacement", "disp"],
            # add other texture types and their possible keywords if needed
        }

        self.texture_configs = {
            self.keyword_order[0]: {"expected_channels": 3, "mode": "RGB"},
            self.keyword_order[1]: {"expected_channels": 3, "mode": "RGB"},
            self.keyword_order[2]: {"expected_channels": 1, "mode": "L"},
            self.keyword_order[3]: {"expected_channels": 1, "mode": "L"},
            self.keyword_order[4]: {"expected_channels": 1, "mode": "L"},
            self.keyword_order[5]: {"expected_channels": 1, "mode": "L"},
            self.keyword_order[6]: {"expected_channels": 1, "mode": "L"},
            # add other texture types if needed
        }

        self.result_tensor = None  # This will hold the final result
        self.material_slices = {}  # This will hold the material type to slice the result tensor by
        self.material_count = 0
        self.lod_cache = {}

    def get_lod_resolution(self, lod, resolution):
        if lod in self.lod_cache:
            return self.lod_cache[lod]
        else:
            self.lod_cache[lod] = torch.nn.functional.interpolate(self.result_tensor[None], size=(resolution, resolution), mode='bilinear', align_corners=True)[0]
            return self.lod_cache[lod]

    def identify_texture_type(self, filename):
        filename_lower = filename.lower()

        # Identify the texture type based on keywords in the filename
        for texture_type, keywords in self.texture_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return texture_type

        raise ValueError(f"Could not identify texture type from filename: {filename}")

    def process_images(self, directory, resolution, dtype=torch.float32):
        textures = {}
        # Loop through all files in the specified directory
        for filename in os.listdir(directory):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff')):  # add other file types if needed
                file_path = os.path.join(directory, filename)

                with Image.open(file_path) as img:
                    texture_type = self.identify_texture_type(filename)
                    if texture_type not in self.texture_configs:
                        raise ValueError(f"Unknown texture type: {texture_type}")

                    img = img.convert(self.texture_configs[texture_type]['mode'])
                    img = img.resize((resolution, resolution), Image.BICUBIC)

                    tensor = to_tensor(img).to(dtype)

                    if tensor.shape[0] != self.texture_configs[texture_type]['expected_channels']:
                        raise ValueError(f"Expected {self.texture_configs[texture_type]['expected_channels']} channels, got {tensor.shape[0]}")

                    textures[texture_type] = tensor

        texture_list = []

        current_index = 0
        for texture_type in self.keyword_order:
            if texture_type in textures and textures[texture_type] is not None:
                texture_list.append(textures[texture_type])
                self.material_slices[texture_type] = (current_index, current_index + textures[texture_type].shape[0])
                current_index += textures[texture_type].shape[0]
                self.material_count += 1

        tensor = torch.cat(texture_list, dim=0)
        # After processing all images, calculate the mean and std for normalization
        self.mean = tensor.mean(dim=[1, 2])  # Mean along the channel dimensions
        min_val = tensor.amin(dim=[1, 2])
        max_val = tensor.amax(dim=[1, 2])
        self.std = torch.max(max_val - self.mean, self.mean - min_val)
        self.result_tensor = self.normalize(tensor)

    def slice_by_material(self, material_type):
        if material_type in self.material_slices:
            start, end = self.material_slices[material_type]
            return self.result_tensor[start:end]
        else:
            raise ValueError(f"Material type '{material_type}' not found in processed tensor.")

    def normalize(self, tensor=None):
        if tensor is None:
            tensor = self.result_tensor
        # This method assumes the input mean and std are tensors with the same number of elements as channels in 'tensor'
        if self.mean.shape[0] != tensor.shape[0] or self.std.shape[0] != tensor.shape[0]:
            raise ValueError("Mean and std length must be equal to the number of channels in the tensor.")

        tensor = normalize(tensor, self.mean, self.std)
        return tensor

    def denormalize(self, tensor=None):
        if tensor is None:
            tensor = self.result_tensor
        # This method assumes the input mean and std are tensors with the same number of elements as channels in 'tensor'
        if self.mean.shape[0] != tensor.shape[0] or self.std.shape[0] != tensor.shape[0]:
            raise ValueError("Mean and std length must be equal to the number of channels in the tensor.")

        return torch.clamp(tensor * self.std[:, None, None] + self.mean[:, None, None], 0, 1)

    def get_material_splits(self):
        return [self.material_slices[t][1]-self.material_slices[t][0] for t in self.keyword_order if t in self.material_slices]

    def split_material(self, tensor=None):
        if tensor is None:
            tensor = self.result_tensor
        materials = torch.split(tensor, self.get_material_splits(), dim=0)
        return materials

    def expand_material(self, materials, channels=3):
        h, w = materials[0].shape[-2:]

        return [mat.expand(channels, h, w) for mat in materials]

    def make_grid(self, tensor=None):
        if tensor is None:
            tensor = self.result_tensor
        materials = self.split_material(self.denormalize(tensor))
        materials = self.expand_material(materials)
        nrows = 1 if len(materials) <= 1 else 2
        #split returns a tuple but make grid expects a list
        # expand_material also converts to list
        return make_grid(materials, nrow=nrows)
