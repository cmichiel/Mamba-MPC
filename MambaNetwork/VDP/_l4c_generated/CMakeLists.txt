
                cmake_minimum_required(VERSION 3.15)
                project(y_expr)
                set(CMAKE_CXX_STANDARD 17)
                set(CMAKE_POSITION_INDEPENDENT_CODE ON)
                
                include_directories(c:/Users/Michiel/anaconda3/envs/run_controller/Lib/site-packages/l4casadi/include)
                link_directories(c:/Users/Michiel/anaconda3/envs/run_controller/Lib/site-packages/l4casadi/lib)

                add_library(y_expr SHARED y_expr.cpp)
                target_link_libraries(y_expr l4casadi)
                
                set_target_properties(y_expr PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY c:/Users/Michiel/OneDrive - TU Eindhoven/Masters/Thesis/MambaNetwork/_l4c_generated
                    RUNTIME_OUTPUT_DIRECTORY c:/Users/Michiel/OneDrive - TU Eindhoven/Masters/Thesis/MambaNetwork/_l4c_generated
                )
                add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

                install(TARGETS y_expr DESTINATION c:/Users/Michiel/OneDrive - TU Eindhoven/Masters/Thesis/MambaNetwork/_l4c_generated)
            