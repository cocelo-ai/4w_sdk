cmake_minimum_required(VERSION 3.14)
project(mode_bindings LANGUAGES CXX)

# ===== Basics =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== Fail fast: required args (NO defaults, NO inference) =====
if(NOT DEFINED PY_MODULE_SRC)
  message(FATAL_ERROR "PY_MODULE_SRC must be defined (absolute path to bindings .cpp)")
endif()
if(NOT IS_ABSOLUTE "${PY_MODULE_SRC}")
  message(FATAL_ERROR "PY_MODULE_SRC must be absolute: ${PY_MODULE_SRC}")
endif()
if(NOT EXISTS "${PY_MODULE_SRC}")
  message(FATAL_ERROR "PY_MODULE_SRC not found: ${PY_MODULE_SRC}")
endif()

if(NOT DEFINED PROJ_ROOT)
  message(FATAL_ERROR "PROJ_ROOT must be defined (absolute project root)")
endif()
if(NOT IS_ABSOLUTE "${PROJ_ROOT}")
  message(FATAL_ERROR "PROJ_ROOT must be absolute: ${PROJ_ROOT}")
endif()
if(NOT IS_DIRECTORY "${PROJ_ROOT}")
  message(FATAL_ERROR "PROJ_ROOT is not a directory: ${PROJ_ROOT}")
endif()

# ===== Output root =====
if(NOT DEFINED PREFIX_DIR)
  set(PREFIX_DIR "${CMAKE_SOURCE_DIR}/dist")
endif()

# ===== Python & pybind11 =====
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(FetchContent)
set(PYBIND11_NEWPYTHON ON)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ===== Paths =====
set(CPP_INCLUDE_DIR "${PROJ_ROOT}/cpp/include")

# ===== Python package output =====
set(PY_PKG_DIR "${PREFIX_DIR}/mode")
file(MAKE_DIRECTORY "${PY_PKG_DIR}")

# ===== Build =====
pybind11_add_module(mode MODULE "${PY_MODULE_SRC}")
target_compile_features(mode PRIVATE cxx_std_17)
target_include_directories(mode PRIVATE "${CPP_INCLUDE_DIR}")


set_target_properties(mode PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  OUTPUT_NAME "mode"
)

# RPATH
if(APPLE)
  set_target_properties(mode PROPERTIES INSTALL_RPATH "@loader_path" BUILD_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(mode PROPERTIES INSTALL_RPATH "\$ORIGIN" BUILD_RPATH "\$ORIGIN")
endif()

# __init__.py
file(GENERATE
  OUTPUT "${PY_PKG_DIR}/__init__.py"
  CONTENT "from .mode import *\n__all__ = ['Mode']\n\n"
)

message("MODE INFO")
message(STATUS "PROJ_ROOT: ${PROJ_ROOT}")
message(STATUS "PY_MODULE_SRC: ${PY_MODULE_SRC}")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "ORT_INC: ${ORT_INC}")
message(STATUS "ORT_LIBDIR: ${ORT_LIBDIR}")
message(STATUS "ONNXRUNTIME_LIB: ${ONNXRUNTIME_LIB}")
message(STATUS "PREFIX_DIR: ${PREFIX_DIR}")
message(STATUS "Python (interp): ${Python3_EXECUTABLE}")
message(STATUS "Output package directory: ${PY_PKG_DIR}")
message("")
