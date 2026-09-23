/* Disable MIT-SHM for this process only, using ordinary XPutImage transfers.
 * No X server settings, host IPC namespace or dataset files are changed.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

/* Some Mesa versions still call XShmPutImage after probing the extension.
 * XImage's local pixel buffer can also be sent through ordinary XPutImage.
 * Opaque pointers avoid requiring X11 development headers in the image.
 */
int XShmPutImage(void *display,
                 unsigned long drawable,
                 void *gc,
                 void *image,
                 int src_x,
                 int src_y,
                 int dest_x,
                 int dest_y,
                 unsigned int width,
                 unsigned int height,
                 int send_event) {
    typedef int (*put_fn)(void *, unsigned long, void *, void *, int, int, int, int, unsigned int,
                          unsigned int);
    static put_fn put_image = NULL;
    (void)send_event;
    if (!put_image) {
        void *xlib = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
        if (xlib) put_image = (put_fn)dlsym(xlib, "XPutImage");
    }
    if (!put_image) return 0;
    put_image(display, drawable, gc, image, src_x, src_y, dest_x, dest_y, width, height);
    return 1;
}

int XQueryExtension(
    void *display, const char *name, int *opcode, int *first_event, int *first_error) {
    if (name && strcmp(name, "MIT-SHM") == 0) {
        if (opcode) *opcode = 0;
        if (first_event) *first_event = 0;
        if (first_error) *first_error = 0;
        return 0;
    }
    typedef int (*query_fn)(void *, const char *, int *, int *, int *);
    query_fn original = (query_fn)dlsym(RTLD_NEXT, "XQueryExtension");
    /* Open3D may load Xlib with RTLD_LOCAL, outside RTLD_NEXT's scope. */
    if (!original) {
        void *xlib = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
        if (xlib) original = (query_fn)dlsym(xlib, "XQueryExtension");
    }
    return original ? original(display, name, opcode, first_event, first_error) : 0;
}

int XShmQueryExtension(void *display) {
    (void)display;
    return 0;
}
int XShmQueryVersion(void *display, int *major, int *minor, int *pixmaps) {
    (void)display;
    if (major) *major = 0;
    if (minor) *minor = 0;
    if (pixmaps) *pixmaps = 0;
    return 0;
}
