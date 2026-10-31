cmake_minimum_required(VERSION 3.20)
project(ExternalConsumer C)

find_package(HeteroAccel REQUIRED CONFIG)

add_executable(external_consumer main.c)
target_link_libraries(external_consumer PRIVATE HeteroAccel::heteroaccel)
