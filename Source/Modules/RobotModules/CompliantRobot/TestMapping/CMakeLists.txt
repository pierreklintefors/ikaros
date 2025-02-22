set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOURCE
	${SOURCE}
	${CMAKE_CURRENT_SOURCE_DIR}/TestMapping.cc
	PARENT_SCOPE
)

# Add Python venv setup
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Check if venv already exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.tensorflow_venv")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/.tensorflow_venv"
        COMMAND ${Python3_EXECUTABLE} -m venv .tensorflow_venv
        COMMAND .tensorflow_venv/bin/pip install --upgrade pip
        COMMAND .tensorflow_venv/bin/pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "For TestMapping module: Creating Python virtual environment for TensorFlow and installing requirements"
    )
    add_custom_target(python_venv
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/.tensorflow_venv"
    )
else()
    add_custom_target(python_venv)
endif()
