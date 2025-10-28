#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash build.sh                 # incremental build

SCRIPT_PATH="$0"
if [[ "$SCRIPT_PATH" == "bash" || "$SCRIPT_PATH" == "-bash" ]]; then
  echo "Please run: bash build.sh"
  exit 1
fi
PROJECT_ROOT="$(cd "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
PREFIX_DIR="${PROJECT_ROOT}/dist"
BUILD_TYPE="${BUILD_TYPE:-Release}"

# --- Pick Python with conda priority ---
if [[ -n "${PYTHON_EXEC:-}" ]]; then
  PYTHON="${PYTHON_EXEC}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
else
  PYTHON="$(command -v python3)"
fi

CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-}"


# allow override via env var
if [[ -n "${PREFIX_DIR_OVERRIDE:-}" ]]; then
  PREFIX_DIR="${PREFIX_DIR_OVERRIDE}"
fi

# Always pass the chosen Python to CMake
PY_ROOT_ARG=""
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PY_ROOT_ARG="-DPython3_ROOT_DIR=${CONDA_PREFIX}"
fi

# ========== onnxpolicy ==========
install -D "${PROJECT_ROOT}/CMakeLists_onnxpolicy.txt" "${PROJECT_ROOT}/onnxpolicy/CMakeLists.txt"

cmake -S "${PROJECT_ROOT}/onnxpolicy" -B "${PROJECT_ROOT}/onnxpolicy" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DPREFIX_DIR="${PROJECT_ROOT}" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -DPROJ_ROOT="${PROJECT_ROOT}" \
  -DPY_MODULE_SRC="${PROJECT_ROOT}/cpp/src/onnxpolicy_bindings.cpp" \
  -DONNXRUNTIME_DIR="${PROJECT_ROOT}/cpp/onnxruntime" \
  ${PY_ROOT_ARG} ${CMAKE_EXTRA_ARGS}

cmake --build "${PROJECT_ROOT}/onnxpolicy" --config "${BUILD_TYPE}" -j

# ========== mode ==========
install -D "${PROJECT_ROOT}/CMakeLists_mode.txt" "${PROJECT_ROOT}/mode/CMakeLists.txt"

cmake -S "${PROJECT_ROOT}/mode" -B "${PROJECT_ROOT}/mode" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DPREFIX_DIR="${PROJECT_ROOT}" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -DPROJ_ROOT="${PROJECT_ROOT}" \
  -DPY_MODULE_SRC="${PROJECT_ROOT}/cpp/src/mode_bindings.cpp" \
  ${PY_ROOT_ARG} ${CMAKE_EXTRA_ARGS}

cmake --build "${PROJECT_ROOT}/mode" --config "${BUILD_TYPE}" -j

# ========== rl ==========
install -D "${PROJECT_ROOT}/CMakeLists_rl.txt" "${PROJECT_ROOT}/rl/CMakeLists.txt"

cmake -S "${PROJECT_ROOT}/rl" -B "${PROJECT_ROOT}/rl" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DPREFIX_DIR="${PROJECT_ROOT}" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -DPROJ_ROOT="${PROJECT_ROOT}" \
  -DPY_MODULE_SRC="${PROJECT_ROOT}/cpp/src/rl_bindings.cpp" \
  ${PY_ROOT_ARG} ${CMAKE_EXTRA_ARGS}

cmake --build "${PROJECT_ROOT}/rl" --config "${BUILD_TYPE}" -j

# ========== robot ==========
install -D "${PROJECT_ROOT}/CMakeLists_robot.txt" "${PROJECT_ROOT}/robot/CMakeLists.txt"

cmake -S "${PROJECT_ROOT}/robot" -B "${PROJECT_ROOT}/robot" -DCMAKE_BUILD_TYPE=Debug \
  -DPREFIX_DIR="${PROJECT_ROOT}" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -DPROJ_ROOT="${PROJECT_ROOT}" \
  -DPY_MODULE_SRC="${PROJECT_ROOT}/cpp/src/robot_bindings.cpp" \
  ${PY_ROOT_ARG} ${CMAKE_EXTRA_ARGS}

cmake --build "${PROJECT_ROOT}/robot" --config "${BUILD_TYPE}" -j


echo "Done !"
