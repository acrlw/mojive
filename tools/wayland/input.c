/* Private Weston 9 acceptance module. Input only enters the owned compositor. */
#define _GNU_SOURCE
#include <libweston/libweston.h>
#include "backend.h"
#include "libweston-internal.h"
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct input {
    struct weston_compositor *compositor;
    struct weston_seat seat;
    struct wl_listener destroy;
    struct wl_event_source *source;
    int fd;
    struct wl_listener frame;
    struct weston_output *output;
    char capture[4096];
};
static void capture_frame(struct wl_listener *listener, void *data) {
    struct input *input = wl_container_of(listener, input, frame);
    struct weston_output *output = input->output;
    struct weston_compositor *compositor = input->compositor;
    wl_list_remove(&input->frame.link);
    wl_list_init(&input->frame.link);
    const int width = output->current_mode->width, height = output->current_mode->height;
    uint32_t *pixels = malloc((size_t)width * height * 4);
    FILE *file = fopen(input->capture, "wb");
    int result = -1;
    if (file && pixels) {
        result = compositor->renderer->read_pixels(output, compositor->read_format,
                                                   pixels, 0, 0, width, height);
        if (result == 0) {
            fprintf(file, "P6\n%d %d\n255\n", width, height);
            for (int y = 0; y < height; ++y) {
                int source_y = compositor->capabilities & WESTON_CAP_CAPTURE_YFLIP ? height-y-1 : y;
                for (int x = 0; x < width; ++x) {
                    uint32_t pixel = pixels[source_y * width + x];
                    unsigned char rgb[] = {pixel >> 16, pixel >> 8, pixel};
                    if (compositor->read_format == PIXMAN_a8b8g8r8 || compositor->read_format == PIXMAN_x8b8g8r8) {
                        rgb[0] = pixel; rgb[2] = pixel >> 16;
                    }
                    if (fwrite(rgb, 1, 3, file) != 3) result = -1;
                }
            }
        }
    }
    if (file && fclose(file) != 0) result = -1;
    free(pixels);
    send(input->fd, result == 0 ? "ok" : "error", result == 0 ? 2 : 5, MSG_NOSIGNAL);
}
static int command(int fd, uint32_t mask, void *data) {
    struct input *input = data;
    char buffer[4096] = {0};
    if (recv(fd, buffer, sizeof(buffer) - 1, 0) <= 0) {
        wl_event_source_remove(input->source);
        input->source = NULL;
        return 0;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    int a, b;
    if (sscanf(buffer, "move %d %d", &a, &b) == 2) {
        notify_motion_absolute(&input->seat, &now, a, b);
        notify_pointer_frame(&input->seat);
    } else if (sscanf(buffer, "button %d %d", &a, &b) == 2) {
        notify_button(&input->seat, &now, a, b);
        notify_pointer_frame(&input->seat);
    } else if (sscanf(buffer, "key %d %d", &a, &b) == 2) {
        notify_key(&input->seat, &now, a, b, STATE_UPDATE_AUTOMATIC);
    } else if (sscanf(buffer, "axis %d %d", &a, &b) == 2) {
        struct weston_pointer_axis_event event = {.axis=a, .value=b};
        notify_axis(&input->seat, &now, &event);
        notify_pointer_frame(&input->seat);
    } else if (sscanf(buffer, "focus %d", &a) == 1) {
        struct weston_pointer *pointer = weston_seat_get_pointer(&input->seat);
        struct weston_surface *surface = a && pointer->focus ? pointer->focus->surface : NULL;
        weston_seat_set_keyboard_focus(&input->seat, surface);
    } else if (strncmp(buffer, "capture ", 8) == 0 && wl_list_empty(&input->frame.link)) {
        if (wl_list_empty(&input->compositor->output_list)) return -1;
        struct weston_output *output = wl_container_of(input->compositor->output_list.next, output, link);
        snprintf(input->capture, sizeof(input->capture), "%s", buffer+8);
        input->output = output;
        wl_signal_add(&output->frame_signal, &input->frame);
        weston_output_schedule_repaint(output);
        return 0;
    } else {
        send(fd, "error", 5, MSG_NOSIGNAL);
        return 0;
    }
    send(fd, "ok", 2, MSG_NOSIGNAL);
    return 0;
}
static void destroy(struct wl_listener *listener, void *data) {
    struct input *input = wl_container_of(listener, input, destroy);
    if (input->source) wl_event_source_remove(input->source);
    wl_list_remove(&input->frame.link);
    weston_seat_release(&input->seat);
    close(input->fd);
    wl_list_remove(&input->destroy.link);
    free(input);
}
WL_EXPORT int wet_module_init(struct weston_compositor *compositor, int *argc, char *argv[]) {
    const char *fd = getenv("MOJIVE_TEST_COMPOSITOR_FD");
    if (!fd) return -1;
    struct input *input = calloc(1, sizeof(*input));
    if (!input) return -1;
    input->compositor = compositor;
    wl_list_init(&input->frame.link);
    input->frame.notify = capture_frame;
    input->fd = atoi(fd);
    weston_seat_init(&input->seat, compositor, "mojive-test-seat");
    weston_seat_init_pointer(&input->seat);
    if (weston_seat_init_keyboard(&input->seat, NULL) < 0) {
        weston_seat_release(&input->seat);
        free(input);
        return -1;
    }
    input->source = wl_event_loop_add_fd(wl_display_get_event_loop(compositor->wl_display), input->fd, WL_EVENT_READABLE, command, input);
    input->destroy.notify = destroy;
    wl_signal_add(&compositor->destroy_signal, &input->destroy);
    return 0;
}
