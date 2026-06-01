from basics.vit import ViT
import torch
import time

def bench_vit(patch_size):
    img_size = 224
    batch_size = 16
    d_model = 384
    n_heads = 6
    n_blocks = 6
    model = ViT(img_size, patch_size, d_model, n_heads, n_blocks).cuda()
    x = torch.randn(batch_size, 3, img_size, img_size).cuda()

    warmup_steps = 5
    timing_steps = 20

    torch.cuda.synchronize()
    for _ in range(warmup_steps):
        _ = model(x)
    torch.cuda.synchronize()

    times = []
    for _ in range(timing_steps):
        start = time.perf_counter()
        _ = model(x)
        end = time.perf_counter()
        times.append((end - start) * 1000) # convert to milliseconds
    torch.cuda.synchronize()
    print(f"mean: {torch.tensor(times).mean():.4f}ms, std: {torch.tensor(times).std():.4f}ms")

def main():
    patch_sizes = [8, 16, 32]
    for patch_size in patch_sizes:
        print(f"Patch size: {patch_size}")
        bench_vit(patch_size)



if __name__ == "__main__":
    main()
