# OpenGL Roadmap – From Basics to 3D Rendering

This roadmap is designed to build a **solid foundation in modern OpenGL (3.3+)**.  
It goes step by step, starting from simple concepts and gradually moving toward rendering, interaction, and shaders.  
The goal is not only to learn how to draw but also to understand the **graphics pipeline** and prepare for **real-time rendering engines**.

---

## 🎯 Objectives
- Understand the OpenGL rendering pipeline (from vertices to pixels).
- Learn how to set up and manage **shaders** (GLSL).
- Practice drawing **2D and 3D objects** with transformations.
- Get comfortable with **buffers** (VBO, VAO, EBO).
- Explore **textures, lighting, and camera controls**.
- Prepare the ground for more advanced topics like **GPU ray tracing** or **engine design**.

---

## 🛠 Prerequisites
- C++ basics (RAII, classes, references).
- Knowledge of vectors, matrices, and basic linear algebra (rotation, scaling, projection).
- A working OpenGL setup with:
  - [GLFW](https://www.glfw.org/) (window/input management)
  - [GLAD](https://glad.dav1d.de/) (OpenGL loader)
  - [glm](https://github.com/g-truc/glm) (math library for transformations)

---

## 🚀 Roadmap

### 1. **Window & Context Setup**
- Use GLFW to create a window.
- Initialize GLAD to load OpenGL functions.
- Handle events (keyboard/mouse).
- Display a simple background color with `glClearColor`.

---

### 2. **Drawing Your First Triangle**
- Create vertex data in a Vertex Buffer Object (VBO).
- Link it to a Vertex Array Object (VAO).
- Write minimal shaders (vertex + fragment).
- Render a triangle.

---

### 3. **Shaders & the Rendering Pipeline**
- Understand how data flows from CPU → GPU.
- Add color to vertices (attribute in the vertex shader).
- Pass data from vertex shader → fragment shader.
- Experiment with uniform variables.

---

### 4. **Transformations**
- Use `glm` for:
  - Translation
  - Rotation
  - Scaling
  - Projection (orthographic vs perspective).
- Render multiple objects with different transforms.

---

### 5. **Textures**
- Load an image (stb_image.h).
- Create a texture object in OpenGL.
- Map texture coordinates (UVs).
- Render textured quads.

---

### 6. **Camera & Interaction**
- Implement a free-moving camera (WASD + mouse).
- Learn about view matrices.
- Use perspective projection for depth.
- Move objects in a scene interactively.

---

### 7. **Lighting & Materials**
- Introduce normals.
- Implement simple Phong shading:
  - Ambient, diffuse, specular.
- Add multiple light sources.
- Experiment with materials (shiny vs dull surfaces).

---

## 📚 Recommended Resources
- [LearnOpenGL](https://learnopengl.com/) – the best modern tutorial.
- [OpenGL Wiki](https://www.khronos.org/opengl/wiki/Main_Page) – reference for functions & concepts.
- [GLSL Sandbox](https://glslsandbox.com/) – experiment with shaders online.

---

## ✅ Next Steps
Once you complete this roadmap, you’ll be ready to:
- Implement **simple 2D/3D engines**.
- Optimize rendering with instancing.
- Move toward **CUDA + OpenGL interop** (next folder).
