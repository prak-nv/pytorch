__global__ void kernel1(Tensor<float, 2> T0, Tensor<float, 2> T1, Tensor<float, 2> T5) {
  float T9[(4 * 4)];
  for(size_t ki255 = 0; ki255 < 4; ++ki255) {
    for(size_t ki256 = 0; ki256 < 4; ++ki256) {
      if (((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki255)) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki256)) < T1.size[1]))) {
        T9[(ki255 * 4) + ki256] = float(0);
      }
    }
  }
  __shared__ float T10[((((ceilDiv(64, 4)) * (ceilDiv(64, 4))) * 4) * 4)];
  for(size_t ki46 = 0; ki46 < (ceilDiv(T0.size[1], 8)); ++ki46) {
    __shared__ float T2[((ceilDiv((64 * 8), 256)) * 256)];
    __shared__ float T3[((ceilDiv((8 * 64), 256)) * 256)];
    float T6[(4 * 4)];
    for(size_t ki225 = 0; ki225 < 4; ++ki225) {
      for(size_t ki226 = 0; ki226 < 4; ++ki226) {
        if (((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki225)) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki226)) < T1.size[1]))) {
          T6[(ki225 * 4) + ki226] = float(0);
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
    for(size_t ki113 = 0; ki113 < 8; ++ki113) {
      float T7[4];
      float T8[4];
      for(size_t ki117 = 0; ki117 < 4; ++ki117) {
        if (((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki117)) < T0.size[0]) && (((ki46 * 8) + ki113) < T0.size[1]))) {
          T7[ki117]
             = T2[((((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki117) * 8) + ki113];
        }
      }
      for(size_t ki151 = 0; ki151 < 4; ++ki151) {
        if (((((ki46 * 8) + ki113) < T1.size[0]) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki151)) < T1.size[1]))) {
          T8[ki151]
             = T3[(ki113 * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki151)];
        }
      }
      for(size_t ki176 = 0; ki176 < 4; ++ki176) {
        for(size_t ki179 = 0; ki179 < 4; ++ki179) {
          float T4[1];
          if ((((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki176)) < T0.size[0]) && (((ki46 * 8) + ki113) < T0.size[1])) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki179)) < T1.size[1]))) {
            T4[0]
              = T7[ki176]
              * T8[ki179];
          }
          if ((((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki176)) < T0.size[0]) && (((ki46 * 8) + ki113) < T0.size[1])) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki179)) < T1.size[1]))) {
            T6[(ki176 * 4) + ki179]
              = T6[(ki176 * 4) + ki179]
              + T4[0];
          }
        }
      }
    }
    for(size_t ki239 = 0; ki239 < 4; ++ki239) {
      for(size_t ki242 = 0; ki242 < 4; ++ki242) {
        if (((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki239)) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki242)) < T1.size[1]))) {
          T9[(ki239 * 4) + ki242]
            = T9[(ki239 * 4) + ki242]
            + T6[(ki239 * 4) + ki242];
        }
      }
    }
    __syncthreads();
  }
  for(size_t ki268 = 0; ki268 < 4; ++ki268) {
    for(size_t ki271 = 0; ki271 < 4; ++ki271) {
      if (((((blockIdx.x * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki268)) < T0.size[0]) && (((blockIdx.y * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki271)) < T1.size[1]))) {
        T10[((((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki268) * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki271)]
           = T9[(ki268 * 4) + ki271];
      }
    }
  }
  __syncthreads();
  for(size_t ki286 = 0; ki286 < 4; ++ki286) {
    for(size_t ki287 = 0; ki287 < 4; ++ki287) {
      if (((((blockIdx.x * 64) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki286)) < T0.size[0]) && (((blockIdx.y * 64) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki287)) < T1.size[1]))) {
        T5[(((blockIdx.x * 64) + (((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki286)) * T5.stride[0]) + ((blockIdx.y * 64) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki287))]
           = T10[((((threadIdx.x / (ceilDiv(64, 4))) * 4) + ki286) * ((ceilDiv(64, 4)) * 4)) + (((threadIdx.x % (ceilDiv(64, 4))) * 4) + ki287)];
      }
    }
  }
}
