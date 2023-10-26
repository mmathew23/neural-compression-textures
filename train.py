import os
import hydra
from omegaconf import DictConfig, OmegaConf
from model.model import Model
from optimizer import get_optimizer_and_lr
from data.dataset import VariableTileDataset, sample_lod
import torch
from torch.utils.data import DataLoader
from utils import get_lod_from_resolution, numel
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def train(config: DictConfig) -> None:
    device = torch.device(config.device)
    dtype = getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32

    dataset = VariableTileDataset(config.dataset.image_path, config.dataset.tile_size, config.dataset.resolution, dtype=dtype, train_len=config.dataset.train_len)
    material = dataset.image
    dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers)
    texture_model = Model(config.model)
    # Following section is needed for inference
    # this tells the model how to split channels for different components
    # and how to denormalize
    texture_model.mean = material.mean
    texture_model.std = material.std
    texture_model.splits = material.get_material_splits()
    ####

    texture_model.to(device=device, dtype=dtype)
    print(f'number of elements {numel(texture_model)}')
    optimizer, scheduler = get_optimizer_and_lr(texture_model, config.trainer.lr_features, config.trainer.lr_mlp, len(dataset), config.trainer.epochs, config.dataloader.batch_size)
    total_steps = config.trainer.epochs * (len(dataset) // config.dataloader.batch_size + (1 if len(dataset) % config.dataloader.batch_size else 0))
    simulated_quantize_cutoff = int(total_steps*0.95)
    simulate_quantization = True
    if simulate_quantization:
        print('simulating quantization')

    lods = get_lod_from_resolution(config.dataset.resolution)
    for epoch in range(config.trainer.epochs):
        batch_progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.trainer.epochs}", leave=True)
        for i, batch in batch_progress:
            optimizer.zero_grad()

            lod = sample_lod(lods)
            pixel_values, coordinates = batch[lod]["pixel_values"], batch[lod]["coordinates"]
            tile_size = batch[lod]["tile_size"]
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            coordinates = coordinates.to(device=device, dtype=torch.long)
            predictions = texture_model(coordinates, tile_size[0], tile_size[0], lod, quantize=simulate_quantization)
            loss = torch.nn.functional.mse_loss(predictions, pixel_values)
            loss.backward()

            batch_progress.set_postfix({"loss": loss.item()})
            optimizer.step()
            scheduler.step()
            if i % 10000 == 0:
                visualize(batch, texture_model, device, dtype, i)

            if (epoch)*len(dataset) + i < simulated_quantize_cutoff:
                texture_model.clamp_values()
            else:
                if simulate_quantization:
                    print('Freezing grids')
                    texture_model.clamp_values()
                    texture_model.quantize_grid_and_freeze()
                    simulate_quantization = False
    visualize(batch, texture_model, device, dtype, -1)
    torch.save(texture_model.cpu(), f"saves/{config.name}.pt")


def visualize(batch, texture_model, device, dtype, idx):
    with torch.no_grad():
        for lod in range(8):
            pixel_values, coordinates = batch[lod]["pixel_values"], batch[lod]["coordinates"]
            tile_size = batch[lod]["tile_size"]
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            coordinates = coordinates.to(device=device, dtype=torch.long)
            predictions = texture_model(coordinates, tile_size[0], tile_size[0], lod)
            # predictions = predictions.cpu()
            materials = []
            for j in range(predictions.shape[0]):
                materials.append(texture_model.make_grid(predictions[j]).cpu())
            grid = make_grid(materials, nrow=4)
            os.makedirs("test_images", exist_ok=True)
            to_pil_image(grid).save(f"test_images/predictions_{idx+1}_{lod}.png")



if __name__ == "__main__":
    train()
