__global__ void CUDAGeneratedKernel(Tensor<float, 2> T0, Tensor<float, 2> T1, Tensor<float, 2> T5) {
  float T9[((ceilDiv(64, 16)) * (ceilDiv(64, 16)))];
  for(size_t ki255 = 0; ki255 < (ceilDiv(64, 16)); ++ki255) {
    for(size_t ki256 = 0; ki256 < (ceilDiv(64, 16)); ++ki256) {
      if (((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki255 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki256 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
        T9[(ki255 * (ceilDiv(64, 16))) + ki256] = float(0);
      }
    }
  }
  __shared__ float T10[(((16 * 16) * (ceilDiv(64, 16))) * (ceilDiv(64, 16)))];
  for(size_t ki46 = 0; ki46 < (ceilDiv(T0.size[1], 8)); ++ki46) {
    __shared__ float T2[((ceilDiv((64 * 8), 256)) * 256)];
    __shared__ float T3[((ceilDiv((8 * 64), 256)) * 256)];
    float T6[((ceilDiv(64, 16)) * (ceilDiv(64, 16)))];
    for(size_t ki225 = 0; ki225 < (ceilDiv(64, 16)); ++ki225) {
      for(size_t ki226 = 0; ki226 < (ceilDiv(64, 16)); ++ki226) {
        if (((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki225 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki226 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
          T6[(ki225 * (ceilDiv(64, 16))) + ki226] = float(0);
        }
      }
    }
    for(size_t ki52 = 0; ki52 < (ceilDiv((64 * 8), 256)); ++ki52) {
      if (((((blockIdx.x * 64) + (((ki52 * 256) + threadIdx.x) / 8)) < T0.size[0]) && (((ki46 * 8) + (((ki52 * 256) + threadIdx.x) % 8)) < T0.size[1]))) {
        T2[((((ki52 * 256) + threadIdx.x) / 8) * 8) + (((ki52 * 256) + threadIdx.x) % 8)]
           = T0[(((blockIdx.x * 64) + (((ki52 * 256) + threadIdx.x) / 8)) * T0.stride[0]) + (((ki46 * 8) + (((ki52 * 256) + threadIdx.x) % 8)) * T0.stride[1])];
      }
    }
    for(size_t ki83 = 0; ki83 < (ceilDiv((8 * 64), 256)); ++ki83) {
      if (((((ki46 * 8) + (((ki83 * 256) + threadIdx.x) / 64)) < T1.size[0]) && (((blockIdx.y * 64) + (((ki83 * 256) + threadIdx.x) % 64)) < T1.size[1]))) {
        T3[((((ki83 * 256) + threadIdx.x) / 64) * 64) + (((ki83 * 256) + threadIdx.x) % 64)]
           = T1[(((ki46 * 8) + (((ki83 * 256) + threadIdx.x) / 64)) * T1.stride[0]) + (((blockIdx.y * 64) + (((ki83 * 256) + threadIdx.x) % 64)) * T1.stride[1])];
      }
    }
    __syncthreads();
    for(size_t ki111 = 0; ki111 < 8; ++ki111) {
      float T7[(ceilDiv(64, 16))];
      float T8[(ceilDiv(64, 16))];
      for(size_t ki116 = 0; ki116 < (ceilDiv(64, 16)); ++ki116) {
        if (((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki116 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki46 * 8) + ki111) < T0.size[1]))) {
          T7[ki116]
             = T2[(((ki116 * 16) + (threadIdx.x / 16)) * 8) + ki111];
        }
      }
      for(size_t ki151 = 0; ki151 < (ceilDiv(64, 16)); ++ki151) {
        if (((((ki46 * 8) + ki111) < T1.size[0]) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki151 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
          T8[ki151]
             = T3[(ki111 * ((ceilDiv(64, 16)) * 16)) + ((ki151 * 16) + (threadIdx.x % 16))];
        }
      }
      for(size_t ki175 = 0; ki175 < (ceilDiv(64, 16)); ++ki175) {
        for(size_t ki179 = 0; ki179 < (ceilDiv(64, 16)); ++ki179) {
          float T4[1];
          if ((((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki175 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki46 * 8) + ki111) < T0.size[1])) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki179 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
            T4[0]
              = T7[ki175]
              * T8[ki179];
          }
          if ((((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki175 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((ki46 * 8) + ki111) < T0.size[1])) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki179 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
            T6[(ki175 * (ceilDiv(64, 16))) + ki179]
              = T6[(ki175 * (ceilDiv(64, 16))) + ki179]
              + T4[0];
          }
        }
      }
    }
    for(size_t ki238 = 0; ki238 < (ceilDiv(64, 16)); ++ki238) {
      for(size_t ki242 = 0; ki242 < (ceilDiv(64, 16)); ++ki242) {
        if (((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki238 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki242 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
          T9[(ki238 * (ceilDiv(64, 16))) + ki242]
            = T9[(ki238 * (ceilDiv(64, 16))) + ki242]
            + T6[(ki238 * (ceilDiv(64, 16))) + ki242];
        }
      }
    }
    __syncthreads();
  }
  for(size_t ki267 = 0; ki267 < (ceilDiv(64, 16)); ++ki267) {
    for(size_t ki271 = 0; ki271 < (ceilDiv(64, 16)); ++ki271) {
      if (((((blockIdx.x * ((ceilDiv(64, 16)) * 16)) + ((ki267 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 16)) * 16)) + ((ki271 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
        T10[(((ki267 * 16) + (threadIdx.x / 16)) * ((ceilDiv(64, 16)) * 16)) + ((ki271 * 16) + (threadIdx.x % 16))]
           = T9[(ki267 * (ceilDiv(64, 16))) + ki271];
      }
    }
  }
  __syncthreads();
  for(size_t ki286 = 0; ki286 < (ceilDiv(64, 16)); ++ki286) {
    for(size_t ki287 = 0; ki287 < (ceilDiv(64, 16)); ++ki287) {
      if (((((blockIdx.x * 64) + ((ki286 * 16) + (threadIdx.x / 16))) < T0.size[0]) && (((blockIdx.y * 64) + ((ki287 * 16) + (threadIdx.x % 16))) < T1.size[1]))) {
        T5[(((blockIdx.x * 64) + ((ki286 * 16) + (threadIdx.x / 16))) * T5.stride[0]) + ((blockIdx.y * 64) + ((ki287 * 16) + (threadIdx.x % 16)))]
           = T10[(((ki286 * 16) + (threadIdx.x / 16)) * ((ceilDiv(64, 16)) * 16)) + ((ki287 * 16) + (threadIdx.x % 16))];
      }
    }
  }
}
