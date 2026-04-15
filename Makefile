NVCC = /usr/local/cuda-12.4/bin/nvcc
FLAGS = -O2 -arch=sm_86 -lm

TARGETS = flash_cuda_slow flash_cuda_flash check_specs

all: $(TARGETS)

flash_cuda_slow: flash_cuda_slow.cu
	$(NVCC) $(FLAGS) $< -o $@

flash_cuda_flash: flash_cuda_flash.cu
	$(NVCC) $(FLAGS) $< -o $@

check_specs: check_specs.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
