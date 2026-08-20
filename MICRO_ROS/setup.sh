alias flash='make -j16 && st-flash --reset write build/*.bin 0x08000000'
alias rebuild_libmicroros='docker run -it --rm -v $(pwd):/project --env MICROROS_LIBRARY_FOLDER=micro_ros_stm32cubemx_utils/microros_static_library microros/micro_ros_static_library_builder:jazzy'
