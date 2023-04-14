from torch.optim import lr_scheduler, Optimizer


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    max_lr: float,
    num_batches: int,
    num_epochs: int,
    warmup_ratio: float,
) -> lr_scheduler:
    total_steps = int(num_batches * num_epochs)

    if scheduler_type == "onecycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy="linear",
        )
    else:
        raise NotImplementedError

    return scheduler
