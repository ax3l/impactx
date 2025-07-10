target_sources(lib
  PRIVATE
    Aperture.cpp
    PlasmaStage.cpp
    Programmable.cpp
    Source.cpp
)

add_subdirectory(diagnostics)
add_subdirectory(transformation)
