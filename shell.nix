{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    llvmPackages.openmp
    raylib
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    glfw
    libGL
    xorg.libX11
    xorg.libXcursor
    xorg.libXrandr
    xorg.libXinerama
    xorg.libXi
  ];

  # ensure compiller can find the libraries
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.vulkan-loader
    pkgs.libGL
    pkgs.raylib
    pkgs.glfw
  ];
}

