# cube.c

`cube.c` is a small 2D physics sandbox written in C using Raylib, GLFW, and Wayland. 

This project isn't built for blazing-fast speed or optimization—it's just a fun personal hobby project exploring physics and rendering from scratch, so performance might be a bit rough around the edges! And yes, this is a bit vibecoded, I am not an expert in C, so do not throw tomatoes at me.

---

## Controls

* **Left Click + Drag** : Throw cubes around
* **Right Click** : Trigger a shockwave
* **N** : Spawn a new cube
* **B** : Toggle inter-cube collisions
* **R** : Reset the scene
* **Spacebar** : Toggle gravity
* **F11** : Toggle borderless fullscreen
* **Hold ESC (3s)** : Quit

---

## How to Compile

First off, you need to install dependencies.

Arch:
```bash
sudo pacman -S raylib glfw vulkan-tools openmp
```

Debian (or Ubuntu):
```bash
sudo apt install libraylib-dev libglfw3-dev vulkan-tools libomp-dev libomp-dev
```

Fedora:
```bash
sudo dnf install raylib-devel glfw-devel vulkan-tools libgomp
```

NixOS:
```bash
# initialize nix shell
nix-shell
# then compile
gcc -fopenmp -O3 cube.c -o cube -lraylib -lglfw -lvulkan -lm -lpthread -ldl -march=native
# if you want cmake, then add it yourself into shell.nix, but ill add it myself soon.
```

### Using CMake

```bash
mkdir build && cd build
cmake ..
make
```

This creates a build directory, you configure it and make it.

### Or if you don't want to use CMake:

```bash
gcc -fopenmp -O3 cube.c -o cube -lraylib -lglfw -lvulkan -lm -lpthread -ldl -march=native
```

This is GNU

---

## Notes

- It's not optimized, so with OpenMP your CPU will probably go 90%
- X11 is **not** supported, but soon will be supported.
- I don't really know what to put here.
- It is a bit buggy on NixOS
