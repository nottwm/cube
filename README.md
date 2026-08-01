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

Compile the program using the following command:

```bash
gcc cube.c -o cube -lraylib -lglfw -lvulkan -lm -lpthread -ldl -lwayland-client
