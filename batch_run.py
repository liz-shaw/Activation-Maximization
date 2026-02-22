import subprocess

model = "vgg16"
layers = [17, 19, 21]
units = [10, 50, 100, 256, 511]
steps = 150

for l in layers:
    for unit in units:
        print(f"\n>>> Running: Layer {l}, Unit {unit}")
        cmd = [
            "python", "main.py",
            "--model", model,
            "--layer", str(l),
            "--unit", str(unit),
            "--steps", str(steps)
        ]
        subprocess.run(cmd)

print("\nAll tasks completed!")