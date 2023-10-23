import hydra
from omegaconf import DictConfig, OmegaConf
from model.model import Model
from dataset import VariableTileDataset
import torch
from torch.utils.data import DataLoader
from utils import get_lod_from_resolution
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image


class Denormalize:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = torch.tensor(mean)[None, :, None, None]
        self.std = torch.tensor(std)[None, :, None, None]

    def __call__(self, tensor):
        return tensor * self.std + self.mean


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def train(config: DictConfig) -> None:
    dataset = VariableTileDataset(config.dataset.image_path, config.dataset.tile_size, config.dataset.resolution)
    dataloader = DataLoader(dataset, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers)
    texture_model = Model(config.model)
    device=torch.device(config.device)
    dtype=getattr(torch, config.dtype) if hasattr(torch, config.dtype) else torch.float32
    texture_model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(texture_model.parameters(), lr=config.trainer.lr)
    denormalize = Denormalize()

    lods = get_lod_from_resolution(config.dataset.resolution)
    for epoch in range(config.trainer.epochs):
        batch_progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.trainer.epochs}", leave=True)
        for i, batch in batch_progress:
            optimizer.zero_grad()
            total_loss = 0
            for lod in lods:
                lod_batch = batch[lod]
                pixel_values, coordinates = lod_batch["pixel_values"], lod_batch["coordinates"]
                tile_size = lod_batch["tile_size"]
                pixel_values = pixel_values.to(device=device, dtype=dtype)
                coordinates = coordinates.to(device=device, dtype=torch.long)
                predictions = texture_model(coordinates, tile_size[0], tile_size[0], lod)
                loss = torch.nn.functional.mse_loss(predictions, pixel_values)
                loss.backward()
                total_loss += loss.item()
            batch_progress.set_postfix({"loss": total_loss})
            optimizer.step()
            if i % 100 == 0:
                # torch.save(texture_model.state_dict(), f"saves/model_{i+1}.pt")
                with torch.no_grad():
                    lod = 0
                    lod_batch = batch[lod]
                    pixel_values, coordinates = lod_batch["pixel_values"], lod_batch["coordinates"]
                    tile_size = lod_batch["tile_size"]
                    pixel_values = pixel_values.to(device=device, dtype=dtype)
                    coordinates = coordinates.to(device=device, dtype=torch.long)
                    predictions = texture_model(coordinates, tile_size[0], tile_size[0], lod)
                    predictions = denormalize(predictions.detach().cpu())
                    grid = make_grid(predictions, nrow=4)
                    to_pil_image(grid).save(f"test_images/predictions_{i+1}.png")


if __name__ == "__main__":
    train()
