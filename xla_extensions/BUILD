load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "orthocache_stream_compact",
    srcs = ["orthocache_stream_compact.cc"],
    hdrs = ["orthocache_stream_compact.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//xla/hlo/ir:hlo",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
    ],
)

xla_cc_test(
    name = "orthocache_stream_compact_test",
    srcs = ["orthocache_stream_compact_test.cc"],
    deps = [
        ":orthocache_stream_compact",
        "//xla:literal",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)
