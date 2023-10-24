from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam


def get_optimizer_and_lr(model, lr_features, lr_mlp, dataset_length, total_epochs, batch_size):
    # Define the optimizer
    optimizer = Adam([
        {'params': model.features.parameters(), 'lr': lr_features},
        {'params': model.mlp.parameters(), 'lr': lr_mlp}
    ])
    total_steps = total_epochs * (dataset_length // batch_size + (1 if dataset_length % batch_size else 0))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.000001)
    return optimizer, scheduler
