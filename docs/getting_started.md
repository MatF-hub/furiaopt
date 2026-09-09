# Getting Started

The first thing you need to do is set `VCPKG_ROOT` to point to your local vcpkg installation (the directory containing `scripts/buildsystems/vcpkg.cmake`). Everything below assumes it's already set.

## Build
Dependencies (Eigen3, spdlog, nlohmann_json, Catch2) are pulled automatically by vcpkg.

    cmake --preset release          # or --preset debug
    cmake --build ../build/release  # binaries land in ../build/release

To also build the test suite, enable it at configure time:

    cmake --preset release -DFURIAOPTIMIZER_BUILD_TESTS=ON
    cmake --build ../build/release
    ../build/release/furiaopt_tests

## Build from VS Code
Open VS Code from a terminal in which `VCPKG_ROOT` is already set.

With the [CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) extension installed, open the `furiaopt` folder directly (not a parent of it): it detects `CMakePresets.json` automatically. Pick a configure preset (status bar, or `Ctrl+Shift+P` > `CMake: Select Configure Preset`), then pick which executable to build/run/debug with `Ctrl+Shift+P` > `CMake: Set Build Target`. Once selected, the status bar's Build/Run/Debug buttons act on that target directly, no terminal needed.

## Running an example
Every example takes the same two flags: a directory containing `config/config.json`, and a directory for its log file.

    ../build/release/rosenbrock_example --config-dir config --logs-output-dir <log_dir>
    ../build/release/qp_trajectory_example --config-dir config --logs-output-dir <log_dir>
    ../build/release/sqp_delta_robot_example --config-dir config --logs-output-dir <log_dir>

## Visualizing a run
Each example has a matching visualizer that parses its log file (`pip install numpy matplotlib pillow`):

    python3 visualizers/rosenbrock.py --log <log_dir>/solver_log.log
    python3 visualizers/qp_2d_trajectory_smoothing.py --log <log_dir>/solver_log.log --out <out_dir>
    python3 visualizers/sqp_delta_robot.py --log <log_dir>/solver_log.log --out <out_dir>

All three accept `--no-show` to save the plots without opening a window.
