#include <heteroaccel/heteroaccel.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("--- HeteroAccel External Consumer Test ---\n");

    const char* ver = ha_version_string();
    if (!ver || strlen(ver) == 0) {
        fprintf(stderr, "Error: null or empty version string\n");
        return 1;
    }
    printf("HeteroAccel Version: %s\n", ver);

    HaRuntime rt = ha_runtime_create();
    if (!rt) {
        fprintf(stderr, "Error: failed to create runtime\n");
        return 2;
    }

    int count = ha_backend_count(rt);
    printf("Discovered backends: %d\n", count);
    if (count < 1) {
        fprintf(stderr, "Error: expected at least 1 backend\n");
        ha_runtime_destroy(rt);
        return 3;
    }

    for (int i = 0; i < count; ++i) {
        HaBackendInfo info;
        if (ha_backend_get(rt, i, &info) == HA_OK) {
            printf("  Backend [%d]: %s (avail=%d, vulkan=%d, cuda=%d)\n",
                   i, info.backend_id, info.available, info.requires_vulkan, info.requires_cuda);
        }
    }

    ha_runtime_destroy(rt);
    printf("External consumer test passed successfully.\n");
    return 0;
}
