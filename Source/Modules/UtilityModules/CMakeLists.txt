set(CMAKE_INCLUDE_CURRENT_DIR ON)
#add_subdirectory(EnergyMeter)
add_subdirectory(Logger)
add_subdirectory(Idle)
add_subdirectory(Constant)
add_subdirectory(Randomizer)

add_subdirectory(CombineFunctions)
add_subdirectory(ReduceFunctions)

add_subdirectory(Scale)
add_subdirectory(Oscillator)
add_subdirectory(FunctionGenerator)
add_subdirectory(Protocol)
add_subdirectory(Print)
add_subdirectory(OutputFile)
add_subdirectory(RotationConverter)
add_subdirectory(Transform)
add_subdirectory(MatrixMultiply)
add_subdirectory(OneHotVector)


add_subdirectory(AmbientLight)

add_subdirectory(SequenceRecorder)


add_subdirectory(LinearSplines)

add_subdirectory(KernelTestModules)

add_subdirectory(PythonScriptCaller)
set(SOURCE ${SOURCE} PARENT_SCOPE)
set(MODULES_INCLUDS ${MODULES_INCLUDS} PARENT_SCOPE)
set(MODULES_LIBS ${MODULES_LIBS} PARENT_SCOPE)

