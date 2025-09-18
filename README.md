# GPU Rendering Playground

Welcome to the **GPU Rendering Playground**!  
This repository is a structured path to learn **OpenGL**, **CUDA**, and their **interoperation** step by step.  
It is designed to build a strong foundation for **real-time graphics**, **GPU simulations**, and **3D engines**.

---

## 🎯 Objectives
- Learn modern OpenGL for **rendering 2D & 3D objects** with shaders.
- Master CUDA for **parallel computing** and GPU acceleration.
- Combine both to create **hybrid pipelines** (simulation + rendering).
- Prepare for projects in **GPU rendering, real-time engines, or ray tracing**.

---

## 📂 Repository Structure

OpenGl/       → OpenGL roadmap & projects
  ├── README.md        → detailed OpenGL roadmap
  └── GameOfLife/      → sample project (cellular automaton)
  └── Step_1		   → first step of the journey
  └── Step_2		   → second step of the journey
Cuda/         → CUDA roadmap & exercises
  └── README.md        → detailed CUDA roadmap
Fusion/       → OpenGL + CUDA interop roadmap
  └── README.md        → detailed interop roadmap

---

## 🚀 Learning Progression – Progress Tracker

Track your progress through **OpenGL**, **CUDA**, and **Fusion (Interop)** steps.  
Check tasks as you complete them!

---

### 🖼 OpenGL (7 steps)

- [x] 🎨 Step 1 — Window & Hello Triangle
- [x] 🎨 Step 2 — Vertex Colors & Shaders
- [x] 🎨 Step 3 — Transformations (Translation, Rotation, Scaling)
- [ ] 🎨 Step 4 — Textures & UV Mapping
- [ ] 🎨 Step 5 — Camera & Interaction (WASD + Mouse)
- [ ] 🎨 Step 6 — Lighting & Materials (Phong Shading)
- [ ] 🎨 Step 7 — 3D Objects & Projection

---

### ⚡ CUDA (6 steps)

- [x] 🚀 Step 1 — Hello CUDA (Threads, Blocks, Grids)
- [x] 🚀 Step 2 — Vector Addition & Memory Transfers
- [x] 🚀 Step 3 — 2D Grid / Game of Life on GPU
- [ ] 🚀 Step 4 — Memory Hierarchy & Optimization
- [ ] 🚀 Step 5 — Advanced Kernels & Synchronization
- [ ] 🚀 Step 6 — Mini-Project: Particle System or Simulation

---

### 🔗 Fusion (3 steps)

- [ ] 🌉 Step 1 — CUDA → OpenGL Texture Interop
- [ ] 🌉 Step 2 — Hybrid Renderer (CUDA computes, OpenGL renders)
- [ ] 🌉 Step 3 — Advanced GPU Pipelines (Streams, Full Simulation Engine)

---

> Note: Each folder (`OpenGl`, `Cuda`, `Fusion`) contains a **README.md** with more details, links, and exercises.
- [OpenGL Roadmap](OpenGl/README.md)
- [CUDA Roadmap](Cuda/README.md)
- [Fusion Roadmap](Fusion/README.md)

---

## 📚 Recommended Resources
- [LearnOpenGL](https://learnopengl.com/) – OpenGL tutorials  
- [CUDA by Example](https://developer.nvidia.com/cuda-example) – CUDA basics  
- [NVIDIA CUDA OpenGL Interop Guide](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GL.html)  
- [GLM Library](https://github.com/g-truc/glm) – math library for OpenGL  

---

## ✅ Tips
- Start with **OpenGL basics**, then CUDA, and finally Fusion (interop).
- Work on small projects at each stage to solidify knowledge.
- Use the roadmaps as **checkpoints**, marking each step completed.
- Keep experimenting with **Game of Life**, particle systems, or small 3D scenes to practice.
