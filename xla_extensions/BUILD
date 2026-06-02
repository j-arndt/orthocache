load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")

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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pattern_matcher",
    ],
)
