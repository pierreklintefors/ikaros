set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOURCE
	${SOURCE}
	${CMAKE_CURRENT_SOURCE_DIR}/MidiInterface.cc
	${CMAKE_CURRENT_SOURCE_DIR}/RtMidi.cpp
	PARENT_SCOPE
)
