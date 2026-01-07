import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

# ============================================
# CONFIGURATION: Change this to profile different models
# ============================================
MODEL_NAME = "resnet34"  # Options: "resnet18", "resnet34", "resnet50", "resnet101"
# ============================================

# Load the selected model
if MODEL_NAME == "resnet18":
    model = models.resnet18()
elif MODEL_NAME == "resnet34":
    model = models.resnet34()
elif MODEL_NAME == "resnet50":
    model = models.resnet50()
elif MODEL_NAME == "resnet101":
    model = models.resnet101()
else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

inputs = torch.randn(5, 3, 224, 224)

print(f"=" * 80)
print(f"Profiling Model: {MODEL_NAME}")
print(f"=" * 80)

# Select whether to profile CUDA activities
profile_CUDA = True

# Move model and inputs to CUDA if available
if profile_CUDA and torch.cuda.is_available():
    model = model.cuda()
    inputs = inputs.cuda()
    print("Running on CUDA")
elif profile_CUDA and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU")
    profile_CUDA = False

if profile_CUDA:
    # Profile multiple iterations for more reliable results
    num_iterations = 12
    print(f"Profiling {num_iterations} iterations...")

    # Configure schedule to profile multiple steps
    # skip_first=2: Skip first 2 iterations (CUDA initialization overhead)
    # wait=1: Wait 1 iteration between cycles
    # warmup=1: 1 warmup iteration per cycle
    # active=3: Profile 3 active iterations per cycle
    # repeat=2: Repeat the cycle 2 times
    # Total: 2 (skip) + (1 wait + 1 warmup + 3 active) × 2 repeats = 12 iterations
    profiler_schedule = schedule(skip_first=2, wait=1, warmup=1, active=3, repeat=2)

    # Save to model-specific directory
    log_dir = f"./log/{MODEL_NAME}"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,  # Enable memory profiling
        schedule=profiler_schedule,  # Add schedule
        on_trace_ready=tensorboard_trace_handler(log_dir),  # TensorBoard integration
    ) as prof:
        for i in range(num_iterations):
            model(inputs)
            prof.step()  # Mark the end of each iteration

    print(f"\nProfiled {num_iterations} iterations - results are averaged across all iterations")
    print(f"TensorBoard trace files saved to: {log_dir}/")
    print(f"\nProfiling Schedule:")
    print(f"  - Steps 0-1: SKIPPED (CUDA initialization)")
    print(f"  - Steps 2-6: Cycle 1 (1 wait + 1 warmup + 3 active) - 3 RECORDED")
    print(f"  - Steps 7-11: Cycle 2 (1 wait + 1 warmup + 3 active) - 3 RECORDED")
    print(f"  Total steps recorded in TensorBoard: 6 (3 per cycle × 2 cycles)")

    # Print memory usage statistics
    print("\n=== Memory Usage (sorted by self_cpu_memory_usage) ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Print CPU-side view
    print("\n=== CPU-Side Metrics (how long CPU managed operations) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Note: Chrome trace export is handled automatically by tensorboard_trace_handler
    print("\n=== Trace Files ===")
    print("Trace files are automatically saved by TensorBoard handler")
    print("Location: ./log/resnet18/")
    print("You can also view individual .pt.trace.json files in chrome://tracing if needed")

    # Print by input shape with memory info
    print("\n=== Grouped by Input Shape (with memory) ===")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Print a simple memory summary
    print("\n=== Memory Summary ===")
    total_cuda_mem = 0
    for event in prof.key_averages():
        # Use cuda_memory_usage attribute (note: not self_cuda_memory_usage)
        if hasattr(event, "cuda_memory_usage") and event.cuda_memory_usage > 0:
            total_cuda_mem += event.cuda_memory_usage
            print(f"{event.key:40s} CUDA Mem: {event.cuda_memory_usage / (1024 * 1024):>8.2f} MB")

    if total_cuda_mem > 0:
        print(f"\nTotal CUDA Memory: {total_cuda_mem / (1024 * 1024):.2f} MB")
    else:
        print("Note: Memory details are available in the table output above (CUDA Mem / Self CUDA Mem columns)")

    # Instructions for viewing in TensorBoard
    print("\n" + "=" * 80)
    print("TENSORBOARD VISUALIZATION")
    print("=" * 80)
    print("To view profiling results in TensorBoard, run:")
    print(f"  tensorboard --logdir=./log")
    print("\nThen open your browser to: http://localhost:6006")
    print("Navigate to the 'PyTorch Profiler' tab to explore the results.")
    print("\nTo compare multiple models:")
    print(f"  1. Run this script with different MODEL_NAME settings (e.g., resnet18, resnet34)")
    print(f"  2. Use 'tensorboard --logdir=./log' to load all runs")
    print(f"  3. In TensorBoard, go to the 'DIFF' tab to compare runs")
    print("=" * 80)
else:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        model(inputs)

    # Print the profiling results
    print(
        prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
    )  # Print top 10 operations by total CPU time
    print(
        prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30)
    )  # Group by input shape and print top 30 operations
