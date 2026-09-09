# Pending unrelated X11 events must not prevent the visibility wait from timing out.
# Keep the pinned submodule unchanged and compile a checked build-tree correction.
set(x11_source "${glfw_SOURCE_DIR}/src/x11_window.c")
file(READ "${x11_source}" x11_code)
set(original [=[static GLFWbool waitForVisibilityNotify(_GLFWwindow* window)
{
    XEvent dummy;
    double timeout = 0.1;

    while (!XCheckTypedWindowEvent(_glfw.x11.display,
                                   window->x11.handle,
                                   VisibilityNotify,
                                   &dummy))
    {
        if (!waitForX11Event(&timeout))
            return GLFW_FALSE;
    }

    return GLFW_TRUE;
}]=])
string(FIND "${x11_code}" "${original}" match)
if(match LESS 0)
    message(FATAL_ERROR "Reconcile the X11 visibility timeout fix with the updated GLFW source")
endif()
set(replacement "${original}")
string(REPLACE "double timeout = 0.1;"
    "double timeout = 0.1;\n    struct pollfd fd = { ConnectionNumber(_glfw.x11.display), POLLIN };"
    replacement "${replacement}")
string(REPLACE "waitForX11Event(&timeout)" "_glfwPollPOSIX(&fd, 1, &timeout)"
    replacement "${replacement}")
string(REPLACE "${original}" "${replacement}" x11_code "${x11_code}")
set(patched_x11 "${CMAKE_BINARY_DIR}/patched/x11_window.c")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/patched")
file(CONFIGURE OUTPUT "${patched_x11}" CONTENT "${x11_code}" @ONLY)
get_target_property(glfw_sources glfw SOURCES)
list(REMOVE_ITEM glfw_sources x11_window.c "${x11_source}")
list(APPEND glfw_sources "${patched_x11}")
set_property(TARGET glfw PROPERTY SOURCES "${glfw_sources}")
target_include_directories(glfw PRIVATE "${glfw_SOURCE_DIR}/src")
