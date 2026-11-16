set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(GazeLock)
add_subdirectory(HeadMirror)
add_subdirectory(KalmanFilter)
add_subdirectory(PIDController)
add_subdirectory(ProximityVergence)

set(SOURCE ${SOURCE} PARENT_SCOPE)
set(MODULES_INCLUDS ${MODULES_INCLUDS} PARENT_SCOPE)
set(MODULES_LIBS ${MODULES_LIBS} PARENT_SCOPE)
