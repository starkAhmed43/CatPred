import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """Warmup + cosine decay scheduler without restarts."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, init_lr: float, max_lr: float, final_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.init_lr = float(init_lr)
        self.max_lr = float(max_lr)
        self.final_lr = float(final_lr)
        self.current_step = 0
        self._step_per_batch = True

        for group in self.optimizer.param_groups:
            group["lr"] = self.init_lr

        super().__init__(optimizer)

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, current_step: int = None):
        if current_step is not None:
            self.current_step = int(current_step)
        else:
            self.current_step += 1

        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            progress = self.current_step / max(1, self.warmup_steps)
            lr = self.init_lr + progress * (self.max_lr - self.init_lr)
        else:
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = (self.current_step - self.warmup_steps) / decay_steps
            progress = min(1.0, max(0.0, progress))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.final_lr + (self.max_lr - self.final_lr) * cosine

        for group in self.optimizer.param_groups:
            group["lr"] = lr


def install_scheduler_patches(lr_scheduler: str = "cosine_warmup") -> None:
    import catpred.utils as utils_module
    import catpred.train.run_training as run_training_module
    from catpred.nn_utils import NoamLR

    utils_module._bench_lr_scheduler_mode = lr_scheduler
    run_training_module._bench_lr_scheduler_mode = lr_scheduler

    if getattr(utils_module, "_bench_scheduler_patched", False):
        return

    original_build_lr_scheduler = utils_module.build_lr_scheduler

    def patched_build_lr_scheduler(optimizer, args, total_epochs=None):
        mode = getattr(utils_module, "_bench_lr_scheduler_mode", "cosine_warmup")
        if mode == "noam":
            return original_build_lr_scheduler(optimizer, args, total_epochs=total_epochs)
        if mode != "cosine_warmup":
            raise ValueError(f"Unsupported lr_scheduler mode: {mode}")

        epochs = int((total_epochs or [args.epochs] * args.num_lrs)[0])
        steps_per_epoch = max(1, int(args.train_data_size // args.batch_size))
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = max(0, int(float(args.warmup_epochs) * steps_per_epoch))

        scheduler = WarmupCosineLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            init_lr=float(args.init_lr),
            max_lr=float(args.max_lr),
            final_lr=float(args.final_lr),
        )
        print(
            "[bench] lr_scheduler=cosine_warmup "
            f"warmup_steps={warmup_steps} total_steps={total_steps} "
            f"init_lr={args.init_lr} max_lr={args.max_lr} final_lr={args.final_lr}",
            flush=True,
        )
        return scheduler

    utils_module.build_lr_scheduler = patched_build_lr_scheduler
    run_training_module.build_lr_scheduler = patched_build_lr_scheduler
    utils_module._bench_scheduler_patched = True
