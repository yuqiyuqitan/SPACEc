# Requirements for installation

## Apple silicon

Currently these files are mostly required for installations on Apple Silicon. The main issue is that `deepcell` requires `tensorflow==2.8`. However, on Apple Silicon we need `tensorflow-macos` and if we want GPU support `tensorflow-metal`. To work around this, the following files install `deepcell` requirements by hand and drop in the right Tensorflow versions. Then `deepcell` can be install with no dependencies and we are golden.

- `requirements-deepcell-mac-arm64_tf28-metal.txt`: For Apple silicon (arm64), if you want `tensorflow==2.8.0`

- `requirements-deepcell-mac-arm64_tf210-metal.txt`: For Apple silicon (arm64), if you want `tensorflow==2.10.0`

- `requirements-deepcell-mac-arm64_tf210.txt`: For Apple silicon (arm64) without support for `tensorflow-metal` (e.g., the macos runner on Github Actions). Uses `tensorflow==2.10.0`. We use this to enable to run our tests on Github Actions.
