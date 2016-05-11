##################################################################################################
# MAKE FILE:  For fMRI_Connectivity.cu 
#
#     target: dependencies
#       [tab] system command
#
#
#####################################################################################################

#set the compiler to NVIDIAS NVCC
CC=/usr/local/cuda/bin/nvcc

INCLUDE=-I/usr/local/cuda/include \
        -I/usr/local/cuda/samples/common/inc

LIBDIR=-L/usr/local/cuda/lib64
LIBS=-lcudart

# CUDA code generation flags - needed for 'atomic-add'
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM30)

EXECUTABLE = connectivity


all: $(EXECUTABLE)

connectivity: fMRI_Main.o GPUDeviceQuery.o nifti1_Read_Write.o fMRI_Sequential.o fMRI_Covariate.o fMRI_Clean.o fMRI_Connectivity.o
	$(CC) $(INCLUDE) $(LIBDIR) fMRI_Main.o GPUDeviceQuery.o nifti1_Read_Write.o fMRI_Sequential.o fMRI_Covariate.o fMRI_Clean.o fMRI_Connectivity.o $(LIBS) -o $@

fMRI_Main.o:  fMRI_Main.cu
	$(CC) $(INCLUDE) $(LIBDIR) -c fMRI_Main.cu $(LIBS)

GPUDeviceQuery.o:  GPUDeviceQuery.cu
	$(CC) $(INCLUDE) $(LIBDIR) -c GPUDeviceQuery.cu $(LIBS)

nifti1_Read_Write.o:  nifti1_Read_Write.cu
	$(CC) $(INCLUDE) $(LIBDIR) -c nifti1_Read_Write.cu $(LIBS)

fMRI_Sequential.o:  fMRI_Sequential.c
	$(CC) $(INCLUDE) $(LIBDIR) -c fMRI_Sequential.c $(LIBS)

fMRI_Covariate.o:  fMRI_Covariate.cu
	$(CC) $(GENCODE_FLAGS) $(INCLUDE) $(LIBDIR) -c fMRI_Covariate.cu $(LIBS)

fMRI_Clean.o:  fMRI_Clean.cu
	$(CC) $(INCLUDE) $(LIBDIR) -c fMRI_Clean.cu $(LIBS)

fMRI_Connectivity.o:  fMRI_Connectivity.cu
	$(CC) $(GENCODE_FLAGS) $(INCLUDE) $(LIBDIR) -c fMRI_Connectivity.cu $(LIBS)


# this is a target:  When the user types "make clean"  this will run
clean:
	rm -rf $(EXECUTABLE) fMRI_Main.o GPUDeviceQuery.o nifti1_Read_Write.o fMRI_Sequential.o fMRI_Covariate.o fMRI_Clean.o fMRI_Connectivity.o












