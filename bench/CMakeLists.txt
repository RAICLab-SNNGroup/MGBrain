
add_executable(sim_vogel sim_vogel.cpp)
target_include_directories(sim_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sim_vogel PUBLIC mgbrain)

add_executable(sim_brunel sim_brunel.cpp)
target_include_directories(sim_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(sim_brunel PUBLIC mgbrain)

# add_executable(sim_brunelp sim_brunelp.cpp)
# target_include_directories(sim_brunelp PUBLIC ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(sim_brunelp PUBLIC mgbrain)

add_executable(part_vogel part_vogel.cpp)
target_include_directories(part_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(part_vogel PUBLIC mgbrain)

add_executable(part_brunel part_brunel.cpp)
target_include_directories(part_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(part_brunel PUBLIC mgbrain)

# add_executable(part_brunelp part_brunelp.cpp)
# target_include_directories(part_brunelp PUBLIC ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(part_brunelp PUBLIC mgbrain)

add_executable(opart_vogel opart_vogel.cpp)
target_include_directories(opart_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(opart_vogel PUBLIC mgbrain)

add_executable(opart_brunel opart_brunel.cpp)
target_include_directories(opart_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(opart_brunel PUBLIC mgbrain)

# add_executable(opart_brunelp opart_brunelp.cpp)
# target_include_directories(opart_brunelp PUBLIC ${CMAKE_SOURCE_DIR}/src)
# target_link_libraries(opart_brunelp PUBLIC mgbrain)

add_executable(test_gen test_gen.cpp)
target_include_directories(test_gen PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_gen PUBLIC mgbrain)

add_executable(test_parser test_parser.cpp)
target_include_directories(test_parser PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_parser PUBLIC mgbrain)

add_executable(test_sim test_sim.cpp)
target_include_directories(test_sim PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sim PUBLIC mgbrain)

add_executable(test_sim_brunel test_sim_brunel.cpp)
# target_compile_options(test_sim_brunel PUBLIC -fsanitize=address)
target_include_directories(test_sim_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sim_brunel PUBLIC mgbrain)

add_executable(test_sim_vogel test_sim_vogel.cpp)
target_include_directories(test_sim_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_sim_vogel PUBLIC mgbrain)

add_executable(test_const test_const.cpp)
target_include_directories(test_const PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_const PUBLIC mgbrain)

add_executable(correct_brunel correct_brunel.cpp)
target_include_directories(correct_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(correct_brunel PUBLIC mgbrain)

add_executable(correct_vogel correct_vogel.cpp)
target_include_directories(correct_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(correct_vogel PUBLIC mgbrain)

add_executable(single_brunel single_brunel.cpp)
target_include_directories(single_brunel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(single_brunel PUBLIC mgbrain)

add_executable(single_vogel single_vogel.cpp)
target_include_directories(single_vogel PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(single_vogel PUBLIC mgbrain)

add_executable(test1 test1.cpp)
target_include_directories(test1 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test1 PUBLIC mgbrain)