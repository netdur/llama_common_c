original_pwd=$(pwd)

llama_cpp_path="$1"
dev_team="$2"

if [ -z "$llama_cpp_path" ]; then
    echo "Usage: $0 [llama_cpp_path] [dev_team]"
    exit 1
fi

# Function to build for a specific platform
build_for_platform() {
  platform=$1
  build_dir="build_${platform}"
  output_dir="${original_pwd}/lib/${platform}"

  rm -rf "$build_dir"
  mkdir "$build_dir"
  cd "$build_dir"

  cmake -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF .. \
    -DCMAKE_BUILD_TYPE=Release -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE="${original_pwd}/ios-arm64.toolchain.cmake" \
    -DPLATFORM="${platform}" -DDEPLOYMENT_TARGET=12 -DENABLE_BITCODE=0 \
    -DENABLE_ARC=0 -DENABLE_VISIBILITY=1 -DENABLE_STRICT_TRY_COMPILE=1

  cmake --build . --config Release

  mkdir -p "${output_dir}"
  if [ "$platform" == "MAC_ARM64" ]; then
    cp "common/Release/libcommon.a" "${output_dir}/libcommon.a"
    cp "Release/libllama.a" "${output_dir}/libllama.a"
    cp "build/ggml.build/Release/libggml.a" "${output_dir}/libggml.a"
  elif [ "$platform" == "OS64" ]; then
    cp "common/Release-iphoneos/libcommon.a" "${output_dir}/libcommon.a"
    cp "Release-iphoneos/libllama.a" "${output_dir}/libllama.a"
    cp "build/ggml.build/Release-iphoneos/libggml.a" "${output_dir}/libggml.a"
  elif [ "$platform" == "SIMULATORARM64" ]; then
    cp "common/Release-iphonesimulator/libcommon.a" "${output_dir}/libcommon.a"
    cp "Release-iphonesimulator/libllama.a" "${output_dir}/libllama.a"
    cp "build/ggml.build/Release-iphonesimulator/libggml.a" "${output_dir}/libggml.a"
  fi

  cd ..
}

# Copy the toolchain file
cp "${original_pwd}/ios-arm64.toolchain.cmake" "${llama_cpp_path}/ios-arm64.toolchain.cmake"

cd "${llama_cpp_path}"

# Build for each platform
build_for_platform "MAC_ARM64"
build_for_platform "OS64"
build_for_platform "SIMULATORARM64"

cd "${original_pwd}"

# Copy headers
rm -rf include
mkdir include
mkdir include/common
cp "${llama_cpp_path}/ggml-backend.h" include/
cp "${llama_cpp_path}/ggml-cuda.cu" include/
cp "${llama_cpp_path}/ggml-cuda.h" include/
cp "${llama_cpp_path}/ggml-metal.h" include/
cp "${llama_cpp_path}/ggml.h" include/
cp "${llama_cpp_path}/llama.h" include/
cp "${llama_cpp_path}/common/common.h" include/common/
cp "${llama_cpp_path}/common/grammar-parser.h" include/common/
cp "${llama_cpp_path}/common/log.h" include/common/
cp "${llama_cpp_path}/common/sampling.h" include/common/

rm -rf assets
mkdir assets
cp "${llama_cpp_path}/build_MAC_ARM64/bin/ggml-metal.metal" assets/

build_llm_library() {
  platform=$1
  build_dir="build_${platform}"

  rm -rf "$build_dir"
  mkdir "$build_dir"
  cd "$build_dir"

  build_executable_flag=""
  if [ "$platform" == "MAC_ARM64" ] && [ -n "$dev_team" ]; then
    build_executable_flag="-DBUILD_EXECUTABLE=ON -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=${dev_team}"
  fi

  cmake .. -DCMAKE_BUILD_TYPE=Release -G Xcode \
    -DCMAKE_TOOLCHAIN_FILE="${original_pwd}/ios-arm64.toolchain.cmake" \
    -DPLATFORM="${platform}" -DDEPLOYMENT_TARGET=12 -DENABLE_BITCODE=0 \
    -DENABLE_ARC=1 -DENABLE_VISIBILITY=1 -DENABLE_STRICT_TRY_COMPILE=1 \
    ${build_executable_flag} -DDEV_TEAM="${dev_team}"

  cmake --build . --config Release --target install

  mkdir "${build_dir}/assets"
  cp "assets/ggml-metal.metal" "${build_dir}/assets/ggml-metal.metal"

  # rm -rf "${build_dir}"
  cd ..
}

rm -rf install
build_llm_library "MAC_ARM64"
build_llm_library "OS64"
build_llm_library "SIMULATORARM64"
