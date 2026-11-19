#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project=$PWD
runtime=${FP_RUNTIME_ROOT:-"$project/.fp-runtime"}
shared_patch=${FP_SHARED_WORKSPACE_PATCH:-"$project/.fp-shared-workspace.patch"}
image=${FP_DOCKER_IMAGE:-ws/isaac_ros:4.6}
gpu=${FP_GPU:-1}

common=(--rm --init --pull=never --user "$(id -u):$(id -g)"
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e MPLCONFIGDIR=/tmp/matplotlib
  -v "$project:/bench" -w /bench)

command=${1:?Usage: run.sh dashboard | run.sh build | run.sh compare-or-live SESSION HEIGHT_M OBJECT_COUNT [SCORE] [LABELS...] | run.sh preview MODE HEIGHT_M OBJECT_COUNT [SCORE] [LABELS...]}
shift
if [[ $command == build ]]; then
  mkdir -p .build
  variants=(native)
  if [[ -f $shared_patch ]]; then
    shared_patch=$(realpath "$shared_patch")
    if [[ ! -d variants/pooled/source/.git ]]; then
      git clone --no-hardlinks upstream/inference variants/pooled/source
      git -C variants/pooled/source apply --check "$shared_patch"
      git -C variants/pooled/source apply "$shared_patch"
    fi
    variants+=(pooled)
  fi
  docker run "${common[@]}" --network none -e FP_CUDA_ARCHITECTURES="${FP_CUDA_ARCHITECTURES:-120}" \
    --entrypoint bash "$image" -ec '
    for variant in "$@"; do
      source=/bench/upstream/inference
      if [[ $variant == pooled ]]; then source=/bench/variants/pooled/source; fi
      cmake -S "$source" -B "/bench/.build/$variant" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="$FP_CUDA_ARCHITECTURES" -DFOUNDATION_POSE_WITH_NVDIFFRAST=OFF
      cmake --build "/bench/.build/$variant" -j4
    done' bash "${variants[@]}"
elif [[ $command == compare || $command == live || $command == preview || $command == dashboard ]]; then
  if [[ ! -x $runtime/runtime/venv/bin/python && ! -L $runtime/runtime/venv/bin/python ]]; then
    printf 'Set FP_RUNTIME_ROOT to the FoundationPose runtime directory described in README.md.\n' >&2
    exit 1
  fi
  runtime=$(realpath "$runtime")
  if [[ $command != dashboard ]]; then
    session=${1:?Session name under results/}
    height=${2:?Positive can height in meters}
    count=${3:?Expected object count}
    if [[ $command == preview ]]; then
      score=${4:-0.25}
      default_detector=rgbd-can
      if [[ -r .models/yoloe-26n-seg-text-320.plan || -n ${FP_YOLO_ENGINE:-} ]]; then default_detector=yolo-trt; fi
      detector=${FP_DETECTOR:-$default_detector}
      warm_detector=${FP_WARM_DETECTOR:-1}
    else
      score=${4:-0.50}
      default_detector=rgbd-can
      if [[ -r $runtime/runtime/detector.py ]]; then default_detector=sdetr; fi
      if [[ -r .models/yoloe-26n-seg-text-320.plan && $default_detector == rgbd-can ]]; then default_detector=yolo-trt; fi
      detector=${FP_DETECTOR:-$default_detector}
      warm_detector=${FP_WARM_DETECTOR:-0}
    fi
    engine_args=()
    if [[ $detector == yolo-trt ]]; then
      engine=${FP_YOLO_ENGINE:-.models/yoloe-26n-seg-text-320.plan}
      engine_args=(--engine "$engine")
      if [[ $warm_detector == 1 ]]; then engine_args+=(--warm-detector); fi
    fi
    if (( $# > 4 )); then
      labels=("${@:5}")
    elif [[ $detector == yolo-trt ]]; then
      labels=(0)
    else
      labels=(22)
    fi
    folder="results/$session"
    if [[ $command != preview ]]; then [[ -r $folder/d435.npz ]]; fi
  fi
  gpu_args=(--gpus "\"device=$gpu\"" --pid host --shm-size 1g
    -v "$runtime:/fp:ro")
  if ! compgen -G "$runtime/runtime/engines/fp16/*.plan" > /dev/null; then
    gpu_args+=(-e FP_POSE_ENGINE_CACHE=/bench/.models/pose-engines)
  fi
  if [[ -d $runtime/original/weights ]]; then
    mkdir -p upstream/original/weights
    gpu_args+=(-v "$runtime/original/weights:/bench/upstream/original/weights:ro")
  fi
  if [[ $command != compare ]]; then
    camera_usb=${FP_CAMERA_USB:-}
    if [[ -z $camera_usb ]]; then
      found=()
      for device in /sys/bus/usb/devices/*; do
        [[ -r $device/idVendor && -r $device/idProduct ]] || continue
        [[ $(< "$device/idVendor") == 8086 && $(< "$device/idProduct") == 0b07 ]] || continue
        printf -v usb_node '/dev/bus/usb/%03d/%03d' "$(cat "$device/busnum")" "$(cat "$device/devnum")"
        found+=("$usb_node")
      done
      if (( ${#found[@]} != 1 )); then
        printf 'Expected one RealSense D435. Set FP_CAMERA_USB to its path from lsusb.\n' >&2
        exit 1
      fi
      camera_usb=${found[0]}
    fi
    camera_sys="/sys$(udevadm info -q path -n "$camera_usb")"
    [[ $(cat "$camera_sys/idVendor") == 8086 && $(cat "$camera_sys/idProduct") == 0b07 ]]
    mapfile -t camera_nodes < <(find "$camera_sys" -regextype posix-extended -type d -regex '.*/video[0-9]+')
    (( ${#camera_nodes[@]} > 0 ))
    gpu_args+=(--group-add "$(stat -c %g "/dev/${camera_nodes[0]##*/}")" --device "$camera_usb" -v /run/udev:/run/udev:ro -e PYTHONPATH=/bench/.camera-deps-py312)
    for node in "${camera_nodes[@]}"; do
      gpu_args+=(--device "/dev/${node##*/}")
    done
  fi
  network=(--network none)
  if [[ $command == preview || $command == dashboard ]]; then
    port=${FP_PREVIEW_PORT:-8765}
    network=(--publish "127.0.0.1:$port:8765")
  fi
  launch() {
    docker run "${common[@]}" "${network[@]}" "${gpu_args[@]}" --entrypoint /fp/runtime/venv/bin/python "$image" "$@"
  }
  if [[ $command == dashboard ]]; then
    printf 'Open http://127.0.0.1:%s in your browser. Press Ctrl+C here to stop.\n' "$port"
    exec docker run "${common[@]}" "${network[@]}" "${gpu_args[@]}" -e FP_MESH_LIBRARY --entrypoint /fp/runtime/venv/bin/python "$image" dashboard.py
  fi
  if [[ $command == preview ]]; then
    printf 'Open http://127.0.0.1:%s in your browser. Press Ctrl+C here to stop.\n' "$port"
    launch preview.py --mode "$session" --height-m "$height" --objects "$count" --detector "$detector" \
      --score "$score" "${engine_args[@]}" --labels "${labels[@]}" --behavior "${FP_PREVIEW_BEHAVIOR:-track}" \
      --redetect "${FP_PREVIEW_REDETECT:-0}" --duration "${FP_PREVIEW_DURATION:-0}"
    exit
  fi
  mkdir -p "$folder"
  launch mesh.py examples/soda-can/coke_can.glb "$folder/scan.ply" --height-m "$height"
  output_prefix=''
  live_args=()
  if [[ $command == live ]]; then
    output_prefix='live-'
    live_args=(--live-frames 10)
  fi
  modes=()
  if [[ -d $runtime/original/weights && -d $runtime/original/mycpp/build ]]; then modes+=(original); fi
  if [[ -f .build/native/libfoundation_pose_nvidia.so ]]; then modes+=(native); fi
  if [[ -f .build/pooled/libfoundation_pose_nvidia.so ]]; then modes+=(pooled pooled-3); fi
  if (( ${#modes[@]} == 0 )); then
    printf 'Build a pose runtime with ./run.sh build before benchmarking.\n' >&2
    exit 1
  fi
  for mode in "${modes[@]}"; do
    printf 'Running %s\n' "$mode"
    launch benchmark.py "$folder/d435.npz" "$folder/$output_prefix$mode.json" --mode "$mode" \
      --height-m "$height" --objects "$count" --detector "$detector" --score "$score" "${engine_args[@]}" "${live_args[@]}" --labels "${labels[@]}" > "$folder/$output_prefix$mode.log" 2>&1
    tail -n 1 "$folder/$output_prefix$mode.log"
  done
  if [[ $command == compare ]]; then launch summarize.py "$folder" --modes "${modes[@]}"; fi
else
  printf 'Unknown command %s\n' "$command" >&2
  exit 2
fi
