__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T1, Tensor<float, 2> T5) {
  float T8[((ceilDiv(64, 16)) * (ceilDiv(64, 16)))];
  if (((((blockIdx.x * 64) + ((((ceilDiv(64, 16)) - 1) * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * 64) + ((((ceilDiv(64, 16)) - 1) * 16) + (threadIdx.x % 16))) < (T0.size[1] * T1.size[1])))) {
    for(size_t ki218 = 0; ki218 < (ceilDiv(64, 16)); ++ki218) {
      for(size_t ki219 = 0; ki219 < (ceilDiv(64, 16)); ++ki219) {
        T8[(ki218 * (ceilDiv(64, 16))) + ki219] = float(0);
      }
    }
  } else {
    for(size_t ki218 = 0; ki218 < (ceilDiv(64, 16)); ++ki218) {
      for(size_t ki219 = 0; ki219 < (ceilDiv(64, 16)); ++ki219) {
        if (((((blockIdx.x * 64) + ((ki218 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * 64) + ((ki219 * 16) + (threadIdx.x % 16))) < (T0.size[1] * T1.size[1])))) {
          T8[(ki218 * (ceilDiv(64, 16))) + ki219] = float(0);
        }
      }
    }
  }
  __shared__ float T9[(((16 * 16) * (ceilDiv(64, 16))) * (ceilDiv(64, 16)))];
  for(size_t ki42 = 0; ki42 < (ceilDiv(T0.size[1], 8)); ++ki42) {
    __shared__ float T2[((ceilDiv((64 * 8), 256)) * 256)];
    __shared__ float T3[((ceilDiv((8 * 64), 256)) * 256)];
    for(size_t ki48 = 0; ki48 < (ceilDiv((64 * 8), 256)); ++ki48) {
      if (((((blockIdx.x * 64) + (((ki48 * 256) + threadIdx.x) / 8)) < T0.size[0]) && (((ki42 * 8) + (((ki48 * 256) + threadIdx.x) % 8)) < T0.size[1]))) {
        T2[((((ki48 * 256) + threadIdx.x) / 8) * 8) + (((ki48 * 256) + threadIdx.x) % 8)]
           = T0[(((blockIdx.x * 64) + (((ki48 * 256) + threadIdx.x) / 8)) * T0.stride[0]) + (((ki42 * 8) + (((ki48 * 256) + threadIdx.x) % 8)) * T0.stride[1])];
      }
    }
    for(size_t ki76 = 0; ki76 < (ceilDiv((8 * 64), 256)); ++ki76) {
      if (((((ki42 * 8) + (((ki76 * 256) + threadIdx.x) / 64)) < T1.size[0]) && (((blockIdx.y * 64) + (((ki76 * 256) + threadIdx.x) % 64)) < T1.size[1]))) {
        T3[((((ki76 * 256) + threadIdx.x) / 64) * 64) + (((ki76 * 256) + threadIdx.x) % 64)]
           = T1[(((ki42 * 8) + (((ki76 * 256) + threadIdx.x) / 64)) * T1.stride[0]) + (((blockIdx.y * 64) + (((ki76 * 256) + threadIdx.x) % 64)) * T1.stride[1])];
      }
    }
    __syncthreads();
    for(size_t ki101 = 0; ki101 < 8; ++ki101) {
      float T6[(ceilDiv(64, 16))];
      float T7[(ceilDiv(64, 16))];
      float T4[((ceilDiv(64, 16)) * (ceilDiv(64, 16)))];
      for(size_t ki112 = 0; ki112 < (ceilDiv(64, 16)); ++ki112) {
        if (((((blockIdx.x * 64) + ((ki112 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki42 * 8) + ki101) < T0.size[1]))) {
          T6[ki112]
             = T2[(((ki112 * 16) + (threadIdx.x / 16)) * 8) + ki101];
        }
      }
      for(size_t ki147 = 0; ki147 < (ceilDiv(64, 16)); ++ki147) {
        if (((((ki42 * 8) + ki101) < T1.size[0]) && (((blockIdx.y * 64) + ((ki147 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
          T7[ki147]
             = T3[(ki101 * 64) + ((ki147 * 16) + (threadIdx.x % 16))];
        }
      }
      if ((((((blockIdx.y * 64) + ((((ceilDiv(64, 16)) - 1) * 16) + (threadIdx.x % 16))) < T1.size[1]) && (((ki42 * 8) + ki101) < T0.size[1])) && (((blockIdx.x * 64) + ((((ceilDiv(64, 16)) - 1) * 16) + (threadIdx.x / 16))) < T0.size[0]))) {
        for(size_t ki171 = 0; ki171 < (ceilDiv(64, 16)); ++ki171) {
          for(size_t ki175 = 0; ki175 < (ceilDiv(64, 16)); ++ki175) {
            T4[(ki171 * (ceilDiv(64, 16))) + ki175]
              = T6[ki171]
              * T7[ki175];
            T8[(ki171 * (ceilDiv(64, 16))) + ki175]
              = T8[(ki171 * (ceilDiv(64, 16))) + ki175]
              + T4[(ki171 * (ceilDiv(64, 16))) + ki175];
          }
        }
      } else {
        for(size_t ki171 = 0; ki171 < (ceilDiv(64, 16)); ++ki171) {
          for(size_t ki175 = 0; ki175 < (ceilDiv(64, 16)); ++ki175) {
            if ((((((blockIdx.x * 64) + ((ki171 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki42 * 8) + ki101) < T0.size[1])) && (((blockIdx.y * 64) + ((ki175 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
              T4[(ki171 * (ceilDiv(64, 16))) + ki175]
                = T6[ki171]
                * T7[ki175];
            }
            if ((((((blockIdx.x * 64) + ((ki171 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki42 * 8) + ki101) < T0.size[1])) && (((blockIdx.y * 64) + ((ki175 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
              T8[(ki171 * (ceilDiv(64, 16))) + ki175]
                = T8[(ki171 * (ceilDiv(64, 16))) + ki175]
                + T4[(ki171 * (ceilDiv(64, 16))) + ki175];
            }
          }
        }
      }
    }
    __syncthreads();
  }
  for(size_t ki231 = 0; ki231 < (ceilDiv(64, 16)); ++ki231) {
    for(size_t ki235 = 0; ki235 < (ceilDiv(64, 16)); ++ki235) {
      if (((((blockIdx.x * 64) + ((ki231 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * 64) + ((ki235 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
        T9[(((ki231 * 16) + (threadIdx.x / 16)) * 64) + ((ki235 * 16) + (threadIdx.x % 16))]
           = T8[(ki231 * (ceilDiv(64, 16))) + ki235];
      }
    }
  }
  __syncthreads();
  for(size_t ki247 = 0; ki247 < (ceilDiv((64 * 64), 256)); ++ki247) {
    if (((((blockIdx.x * 64) + (((ki247 * 256) + threadIdx.x) / 64)) < T0.size[0]) && (((blockIdx.y * 64) + (((ki247 * 256) + threadIdx.x) % 64)) < T1.size[1]))) {
      T5[(((blockIdx.x * 64) + (((ki247 * 256) + threadIdx.x) / 64)) * T5.stride[0]) + ((blockIdx.y * 64) + (((ki247 * 256) + threadIdx.x) % 64))]
         = T9[((((ki247 * 256) + threadIdx.x) / 64) * 64) + (((ki247 * 256) + threadIdx.x) % 64)];
    }
  }
}
