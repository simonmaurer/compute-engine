load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

load("@bazel_skylib//lib:selects.bzl", "selects")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64e",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64e",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
        ":macos_arm64e",
    ],
)

cc_binary(
    name = "lce_benchmark_model",
    srcs = [
        "lce_benchmark_main.cc",
    ],
    copts = ["-Wall"] + tflite_copts(),
    linkopts = tflite_linkopts() + select({
        "@org_tensorflow//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//larq_compute_engine/tflite/kernels:lce_op_kernels",
        "@org_tensorflow//tensorflow/lite/tools:logging",
        "@org_tensorflow//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)
