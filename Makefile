
CXX=g++
CFLAGS = -Wall -O3
LDLIBS = -lm -lopenblas -llapacke -L./vlfeat-0.9.21/bin/glnxa64 -lvl
CPPFLAGS = -I./vlfeat-0.9.21

DIRS = texture utility svm .
OBJDIR = output

SRCS := $(wildcard $(addsuffix /*.cpp,$(DIRS)))
OBJS = $(addprefix $(OBJDIR)/,$(notdir $(patsubst %.cpp,%.o,$(SRCS))))

vpath %.cpp $(DIRS)




all: create_output_dir texture_classification_scheme

texture_classification_scheme: $(OBJS)
		$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJDIR)/%.o : %.cpp
		$(CXX) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

create_output_dir:
		mkdir -p ${OBJDIR}

clean:
		rm -rf $(OBJDIR) *.model *.ppm *.txt texture_classification_scheme




.PHONY: all clean
