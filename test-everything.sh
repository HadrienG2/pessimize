#!/bin/bash
set -euo pipefail

### CROSS-COMPILED PLATFORMS ###
# TODO: Try to run tests via qemu

# Arguments: target rustflags extra_features +nightly
function cross_build_base() {
    for subcommand in 'clippy' 'clippy --tests' 'build' 'build --tests' doc; do
        command="cargo $4 ${subcommand} --target=$1 $3"
        printf "\nRUSTFLAGS=\"$2\" $command\n"
        RUSTFLAGS=\"$2\" $($command)
    done
}

# Arguments: target rustflags
# FIXME: Merge into one and try nightly everywhere once appropriate toolchains
#        and targets are installed
function cross_build() {
    cross_build_base "$1" "$2" "" ''
    cross_build_base "$1" "$2" "--features=safe_arch" ''
}
#
function cross_nightly_build() {
    cross_build_base "$1" "$2" "--features=nightly" '+nightly'
    cross_build_base "$1" "$2" "--features=nightly,safe_arch" '+nightly'
}

for rustflags in '' '-C target-feature=-neon' '-C target-feature=+neon'; do
    cross_build aarch64-unknown-linux-gnu "$rustflags"
done

# FIXME: These tests build but do not link at the moment
set +e
for rustflags in '' '-C target-feature=-vfp2' '-C target-feature=+vfp2'; do
    cross_build arm-unknown-linux-gnueabi "$rustflags"
done
set -e

# FIXME: These tests build but do not link at the moment
set +e
for rustflags in '' \
                 '-C target-feature=+sse' \
                 '-C target-feature=+sse -C target-feature=+sse2'; do
    cross_build i586-unknown-linux-gnu "$rustflags"
done
set -e

# TODO: Would like to test riscv32, but no cross-compiler available

for rustflags in '' \
                 '-C target-feature=-f -C target-feature=-d' \
                 '-C target-feature=+f -C target-feature=-d' \
                 '-C target-feature=-f -C target-feature=+d' \
                 '-C target-feature=+f -C target-feature=+d'; do
    cross_build riscv64gc-unknown-linux-gnu "$rustflags"
done

for rustflags in '-C target-feature=+avx512f' \
                 '-C target-feature=+avx512f -C target-feature=+avx512bw' \
                 '-C target-feature=+avx512f -C target-feature=+avx512vl' \
                 '-C target-feature=+avx512f -C target-feature=+avx512bf16' \
                 '-C target-feature=+avx512f -C target-feature=+avx512bf16 -C target-feature=+avx512vl'; do
    cross_nightly_build x86_64-unknown-linux-gnu "$rustflags"
done

### NATIVE PLATFORM ###

for rustflags in '' \
                 '-C target-feature=+avx' \
                 '-C target-feature=+avx -C target-feature=+avx2'; do
    RUSTFLAGS="${rustflags}" cargo build
    RUSTFLAGS="${rustflags}" cargo build --features=safe_arch
    for config in '' '--release -- --include-ignored'; do
            RUSTFLAGS="${rustflags}" cargo test ${config}
            RUSTFLAGS="${rustflags}" cargo test ${config}
            RUSTFLAGS="${rustflags}" cargo +nightly test --features=nightly ${config}
    done
done
