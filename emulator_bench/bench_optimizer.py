import torch


def install_optimizer_patches(fused_mode: str = "auto") -> None:
    import catpred.train.run_training as run_training_module
    import catpred.utils as utils_module

    utils_module._bench_optimizer_fused_mode = fused_mode
    run_training_module._bench_optimizer_fused_mode = fused_mode

    if getattr(utils_module, "_bench_optimizer_patched", False):
        return

    original_build_optimizer = utils_module.build_optimizer

    def patched_build_optimizer(model, args):
        params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]
        mode = getattr(utils_module, "_bench_optimizer_fused_mode", "auto")
        device_str = str(getattr(args, "device", "cpu")).lower()
        want_cuda = "cuda" in device_str and torch.cuda.is_available()

        use_fused = False
        if mode == "on":
            use_fused = want_cuda
        elif mode == "auto":
            use_fused = want_cuda
        elif mode == "off":
            use_fused = False
        else:
            raise ValueError(f"Unsupported optimizer_fused mode: {mode}")

        if use_fused:
            try:
                optimizer = torch.optim.Adam(params, fused=True)
                optimizer._bench_fused = True
                print(f"[bench] optimizer_fused={mode} -> using fused Adam", flush=True)
                return optimizer
            except Exception:
                if mode == "on":
                    raise
                print(f"[bench] optimizer_fused={mode} -> fused Adam unavailable, falling back to standard Adam", flush=True)

        optimizer = original_build_optimizer(model, args)
        optimizer._bench_fused = False
        if mode == "off":
            print("[bench] optimizer_fused=off -> using standard Adam", flush=True)
        elif not want_cuda:
            print(f"[bench] optimizer_fused={mode} -> CUDA unavailable, using standard Adam", flush=True)
        return optimizer

    utils_module.build_optimizer = patched_build_optimizer
    run_training_module.build_optimizer = patched_build_optimizer
    utils_module._bench_optimizer_patched = True
