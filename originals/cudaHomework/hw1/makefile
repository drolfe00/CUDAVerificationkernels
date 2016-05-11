CC=/usr/local/cuda/bin/nvcc
INCLUDE=-I/usr/local/cuda/include \

LIBDIR=-L/usr/local/cuda/lib64
LIBS=-lcudart

SOURCE=driver.cu
EXECUTABLE=addmul

$(EXECUTABLE): $(SOURCE)
	$(CC) $(INCLUDE) $(LIBDIR) $< -o $@ $(LIBS)

clean:
	rm -f $(EXECUTABLE)
