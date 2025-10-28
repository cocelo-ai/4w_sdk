cmake_minimum_required(VERSION 3.14)
project(onnxpolicy_bindings LANGUAGES CXX)

# ===== Options =====
set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# PREFIX_DIR can be overridden via -DPREFIX_DIR=...
if(NOT DEFINED PREFIX_DIR)
  set(PREFIX_DIR "${CMAKE_SOURCE_DIR}/dist")
endif()

# Find Python interpreter (we only require interpreter here)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Fetch pybind11 (FetchContent)
include(FetchContent)
set(PYBIND11_NEWPYTHON ON)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# ===== Paths =====
if(NOT DEFINED PROJ_ROOT)
  set(PROJ_ROOT "${CMAKE_SOURCE_DIR}")
endif()
if(NOT DEFINED CPP_INCLUDE_DIR)
  set(CPP_INCLUDE_DIR "${PROJ_ROOT}/cpp/include")
endif()

# Resolve ONNXRUNTIME root/include/lib
if(DEFINED ONNXRUNTIME_DIR)
  set(_ORT_ROOT "${ONNXRUNTIME_DIR}")
elseif(DEFINED ENV{ONNXRUNTIME_DIR})
  set(_ORT_ROOT "$ENV{ONNXRUNTIME_DIR}")
else()
  set(_ORT_ROOT "${PROJ_ROOT}/cpp/onnxruntime")
endif()

if(DEFINED ONNXRUNTIME_INCLUDE_DIR)
  set(_ORT_INC "${ONNXRUNTIME_INCLUDE_DIR}")
else()
  set(_ORT_INC "${_ORT_ROOT}/include")
endif()

if(DEFINED ONNXRUNTIME_LIB_DIR)
  set(_ORT_LIBDIR "${ONNXRUNTIME_LIB_DIR}")
else()
  set(_ORT_LIBDIR "${_ORT_ROOT}/lib")
endif()

# sanity check
if(NOT EXISTS "${_ORT_INC}" OR NOT EXISTS "${_ORT_LIBDIR}")
  message(FATAL_ERROR "onnxruntime include/lib not found. Tried:\n  include: ${_ORT_INC}\n  lib: ${_ORT_LIBDIR}\nHint: pass -DONNXRUNTIME_DIR=/path/to/onnxruntime or set ONNXRUNTIME_DIR env or set ONNXRUNTIME_INCLUDE_DIR/ONNXRUNTIME_LIB_DIR.")
endif()

# canonical variable for messages
set(ONNXRUNTIME_DIR "${_ORT_ROOT}")

# ===== Python package output folder =====
set(PY_PKG_DIR "${PREFIX_DIR}/onnxpolicy")
file(MAKE_DIRECTORY "${PY_PKG_DIR}")

# ===== binding source =====
if(NOT DEFINED PY_MODULE_SRC)
  set(PY_MODULE_SRC "${PROJ_ROOT}/src/onnxpolicy_binding.cpp")
endif()
if(NOT EXISTS "${PY_MODULE_SRC}")
  message(FATAL_ERROR "Expected binding source not found: ${PY_MODULE_SRC}")
endif()

# Find onnxruntime lib
find_library(ONNXRUNTIME_LIB NAMES onnxruntime libonnxruntime HINTS "${_ORT_LIBDIR}" REQUIRED)
file(GLOB _cand "${_ORT_LIBDIR}/libonnxruntime.so*")
if(_cand)
  list(GET _cand 0 ONNXRUNTIME_DLL)
else()
  set(ONNXRUNTIME_DLL "${_ORT_LIBDIR}/libonnxruntime.so")
endif()

# ===== create the Python extension target =====
# Your C++ should have: PYBIND11_MODULE(onnxpolicy, m)
pybind11_add_module(_onnxpolicy MODULE "${PY_MODULE_SRC}")
target_compile_features(_onnxpolicy PRIVATE cxx_std_17)
target_include_directories(_onnxpolicy PRIVATE "${CPP_INCLUDE_DIR}" "${_ORT_INC}")
target_link_libraries(_onnxpolicy PRIVATE "${ONNXRUNTIME_LIB}")

# Name the produced file "onnxpolicy*.so" (no leading "lib" etc.)
set_target_properties(_onnxpolicy PROPERTIES
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${PY_PKG_DIR}"
  OUTPUT_NAME "onnxpolicy"
)

# RPATH so that the extension looks for shared libs next to itself
if(APPLE)
  set_target_properties(_onnxpolicy PROPERTIES INSTALL_RPATH "@loader_path" BUILD_RPATH "@loader_path")
elseif(UNIX)
  set_target_properties(_onnxpolicy PROPERTIES INSTALL_RPATH "\$ORIGIN" BUILD_RPATH "\$ORIGIN")
endif()

# Create __init__.py and VERSION in package dir safely
file(GENERATE
  OUTPUT "${PY_PKG_DIR}/__init__.py"
  CONTENT "from .onnxpolicy import MLPPolicy, LSTMPolicy\n__all__ = ['MLPPolicy','LSTMPolicy']\n"
)
file(GENERATE
  OUTPUT "${PY_PKG_DIR}/VERSION"
  CONTENT "0.1.0\n"
)

# Copy onnxruntime shared lib next to the module
if(EXISTS "${ONNXRUNTIME_DLL}")
  add_custom_command(TARGET _onnxpolicy POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_DLL}" "${PY_PKG_DIR}"
  )
endif()

message("ONNXPOLICY INFO")
message(STATUS "PREFIX_DIR: ${PREFIX_DIR}")
message(STATUS "Python (interp): ${Python3_EXECUTABLE}")
message(STATUS "onnxruntime root: ${ONNXRUNTIME_DIR}")
message(STATUS "onnxruntime include: ${_ORT_INC}")
message(STATUS "onnxruntime libdir: ${_ORT_LIBDIR}")
message("")