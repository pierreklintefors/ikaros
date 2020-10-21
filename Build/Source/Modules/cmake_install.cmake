# Install script for directory: /Users/studnet2/Desktop/ikaros - forked/ikaros/Source/Modules

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/Library/Developer/CommandLineTools/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/ANN/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/BrainModels/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/CodingModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/ControlModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/EnvironmentModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/Examples/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/IOModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/LearningModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/RobotModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/UtilityModules/cmake_install.cmake")
  include("/Users/studnet2/Desktop/ikaros - forked/ikaros/Build/Source/Modules/VisionModules/cmake_install.cmake")

endif()

