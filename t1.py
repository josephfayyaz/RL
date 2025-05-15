import os
os.environ["MUJOCO_GL"] = "egl"

from mujoco_py import load_model_from_path, MjSim, MjRenderContextOffscreen
from PIL import Image
import numpy as np

print("MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

model = load_model_from_path("/home/joseph/.mujoco/mujoco210/model/humanoid.xml")
sim = MjSim(model)
viewer = MjRenderContextOffscreen(sim)
sim.add_render_context(viewer)

for _ in range(10):  # Reduced for testing
    sim.step()
    viewer.render(width=640, height=480, camera_id=-1)
    rgb = viewer.read_pixels(640, 480, depth=False)
    img = Image.fromarray(np.flipud(rgb))  # Flip vertically
    img.save("frame.png")

print("parastoo")