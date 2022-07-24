#!/bin/bash
set -euo pipefail

### CROSS-COMPILED PLATFORMS ###
# TODO: Try to run tests via qemu

function cross_build() {
    command="cargo build --tests --target=$1"
    printf "\nRUSTFLAGS=\"$2\" $command\n"
    RUSTFLAGS=\"$2\" $($command)
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

### NATIVE PLATFORM ###

for config in '' '--release -- --include-ignored'; do
    for rustflags in '' \
                     '-C target-feature=+avx' \
                     '-C target-feature=+avx -C target-feature=+avx2'; do
        RUSTFLAGS="${rustflags}" cargo test ${config}
    done
done