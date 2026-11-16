set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(Arousal)
add_subdirectory(BehaviorSelector)
add_subdirectory(Blinker)
add_subdirectory(GlanceGenerator)
add_subdirectory(KeyframeAnimator)
add_subdirectory(OscillationDetector)

set(SOURCE ${SOURCE} PARENT_SCOPE)
set(MODULES_INCLUDS ${MODULES_INCLUDS} PARENT_SCOPE)
set(MODULES_LIBS ${MODULES_LIBS} PARENT_SCOPE)
