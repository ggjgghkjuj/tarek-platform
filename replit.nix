{ pkgs }: {
  deps = [
    pkgs.cmake
    pkgs.gcc
    pkgs.pkg-config
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.libGL
    pkgs.libGLU
  ];
}