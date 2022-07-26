#!/bin/bash
set -euo pipefail

### CROSS-COMPILED PLATFORMS ###
# TODO: Introduce three tiers:
#       1. clippy + build crate (can be done even on a -none target)
#       2. clippy + build crate & tests (can be done on a cross -linux target))
#       3. clippy + build crate + run tests (native target)
# TODO: Try to run tests via qemu

# Runs a cargo command with certain RUSTFLAGS, echoing it beforehand
# Arguments: rustflags command
function cargo_echo() {
    command="cargo $2"
    printf "\nRUSTFLAGS=\"$1\" $command\n"
    RUSTFLAGS="$1" $command
}

# Arguments: target rustflags extra_features +nightly
function cross_build_base() {
    for subcommand in 'clippy' 'clippy --tests' 'build' 'build --tests' doc; do
        cargo_echo "$2" "$4 ${subcommand} --target=$1 $3"
    done
}

# Arguments: target rustflags
# FIXME: Merge those to try nightly everywhere once appropriate toolchains
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
    for subcommand in clippy build; do
        for features in '' '--features=safe_arch'; do
            cargo_echo "$rustflags" "$subcommand $features"
        done
    done
    for config in '' '--release -- --include-ignored'; do
        for op in 'test' '+nightly test --features=nightly' '+nightly test --features=default_impl'; do
            cargo_echo "$rustflags" "$op $config"
        done
    done
done
