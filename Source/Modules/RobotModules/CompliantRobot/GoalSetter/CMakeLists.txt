set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Dynamixel SDK
find_package(DYNAMIXEL)

# if all libraries found add source file, include dirs and libraries
if(DYNAMIXEL_LIB_FOUND)

	set(SOURCE
	${SOURCE}
	${CMAKE_CURRENT_SOURCE_DIR}/GoalSetter.cc
	PARENT_SCOPE
)

	set(MODULES_INCLUDS
	${MODULES_INCLUDS}
	${DYNAMIXEL_INCLUDE_DIRS}
	PARENT_SCOPE
   	)
   	set(MODULES_LIBS
	${MODULES_LIBS}
	${DYNAMIXEL_LIBRARIES}
	PARENT_SCOPE
    ) 

else()
	message(STATUS "Ikaros: Dynamixel libs not found. GoalSetter module is not included into the build. Visit wiki of github page for information of module dependencies." )
endif()


