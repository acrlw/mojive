/* An external wl_data_device source for real file-drop acceptance. */
#define _GNU_SOURCE
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <wayland-client.h>
#include <linux/input-event-codes.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static struct wl_compositor *compositor;
static struct wl_shm *shm;
static struct wl_seat *seat;
static struct wl_data_device_manager *manager;
static struct wl_data_device *device;
static GLFWwindow *window;
static const char *uri;
static int finished;

static void source_target(void *data, struct wl_data_source *source, const char *mime) {}
static void source_send(void *data, struct wl_data_source *source, const char *mime, int fd) {
    FILE *file = fdopen(fd, "w");
    if (file) { fprintf(file, "%s\r\n", uri); fclose(file); }
    else close(fd);
}
static void source_cancel(void *data, struct wl_data_source *source) { finished = 1; }
static void source_drop(void *data, struct wl_data_source *source) {}
static void source_finish(void *data, struct wl_data_source *source) { finished = 1; }
static void source_action(void *data, struct wl_data_source *source, uint32_t action) {}
static const struct wl_data_source_listener source_listener = {
    source_target, source_send, source_cancel, source_drop, source_finish, source_action
};
static void pointer_enter(void *data, struct wl_pointer *pointer, uint32_t serial,
                          struct wl_surface *surface, wl_fixed_t x, wl_fixed_t y) {}
static void pointer_leave(void *data, struct wl_pointer *pointer, uint32_t serial, struct wl_surface *surface) {}
static void pointer_motion(void *data, struct wl_pointer *pointer, uint32_t time, wl_fixed_t x, wl_fixed_t y) {}
static void pointer_axis(void *data, struct wl_pointer *pointer, uint32_t time, uint32_t axis, wl_fixed_t value) {}
static void pointer_button(void *data, struct wl_pointer *pointer, uint32_t serial, uint32_t time, uint32_t button, uint32_t state) {
    if (button != BTN_LEFT || state != WL_POINTER_BUTTON_STATE_PRESSED) return;
    struct wl_data_source *source = wl_data_device_manager_create_data_source(manager);
    wl_data_source_add_listener(source, &source_listener, NULL);
    wl_data_source_offer(source, "text/uri-list");
    wl_data_source_set_actions(source, WL_DATA_DEVICE_MANAGER_DND_ACTION_COPY);
    wl_data_device_start_drag(device, source, glfwGetWaylandWindow(window), NULL, serial);
    /* Unmap the origin to expose the target while the independent data source
       retains ownership until the destination finishes reading its pipe. */
    glfwHideWindow(window);
    puts("dragging"); fflush(stdout);
}
static const struct wl_pointer_listener pointer_listener = {
    .enter = pointer_enter, .leave = pointer_leave, .motion = pointer_motion,
    .button = pointer_button, .axis = pointer_axis
};
static void global(void *data, struct wl_registry *registry, uint32_t name, const char *interface, uint32_t version) {
    if (!strcmp(interface, "wl_compositor")) compositor = wl_registry_bind(registry, name, &wl_compositor_interface, 1);
    else if (!strcmp(interface, "wl_shm")) shm = wl_registry_bind(registry, name, &wl_shm_interface, 1);
    else if (!strcmp(interface, "wl_seat")) seat = wl_registry_bind(registry, name, &wl_seat_interface, 1);
    else if (!strcmp(interface, "wl_data_device_manager") && version >= 3) manager = wl_registry_bind(registry, name, &wl_data_device_manager_interface, 3);
}
static void global_remove(void *data, struct wl_registry *registry, uint32_t name) {}
static const struct wl_registry_listener registry_listener = {global, global_remove};

int main(int argc, char **argv) {
    if (argc != 2 || !glfwInit()) return 1;
    uri = argv[1];
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(1400, 1000, "Mojive file drag source", NULL, NULL);
    if (!window || glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) return 2;
    struct wl_display *display = glfwGetWaylandDisplay();
    struct wl_registry *registry = wl_display_get_registry(display);
    wl_registry_add_listener(registry, &registry_listener, NULL);
    wl_display_roundtrip(display);
    if (!compositor || !shm || !seat || !manager) return 3;
    device = wl_data_device_manager_get_data_device(manager, seat);
    struct wl_pointer *pointer = wl_seat_get_pointer(seat);
    wl_pointer_add_listener(pointer, &pointer_listener, NULL);
    glfwShowWindow(window);
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    size_t bytes = (size_t)width * height * 4;
    int fd = memfd_create("mojive-drag", 0);
    if (fd < 0 || ftruncate(fd, bytes) < 0) return 4;
    uint32_t *pixels = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (pixels == MAP_FAILED) return 5;
    for (size_t i = 0; i < bytes/4; ++i) pixels[i] = 0xff406080;
    struct wl_shm_pool *pool = wl_shm_create_pool(shm, fd, bytes);
    struct wl_buffer *buffer = wl_shm_pool_create_buffer(pool, 0, width, height, width*4, WL_SHM_FORMAT_XRGB8888);
    struct wl_surface *surface = glfwGetWaylandWindow(window);
    wl_surface_attach(surface, buffer, 0, 0);
    wl_surface_damage(surface, 0, 0, width, height);
    wl_surface_commit(surface);
    wl_display_roundtrip(display);
    puts("ready"); fflush(stdout);
    while (!finished && !glfwWindowShouldClose(window)) glfwWaitEventsTimeout(0.05);
    wl_buffer_destroy(buffer);
    wl_shm_pool_destroy(pool);
    munmap(pixels, bytes);
    close(fd);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
