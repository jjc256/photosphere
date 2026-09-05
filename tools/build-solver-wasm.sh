#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/../solver"
rustup run stable cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/photosphere_solver.wasm ../js/vendor/photosphere_solver.wasm
