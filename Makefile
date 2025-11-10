# Makefile for Machine_Learning_Tutorial

CC := gcc
CFLAGS := -Wall -Wextra -Wpedantic -g
SRCDIR := src
BINDIR := bin
TARGET := $(BINDIR)/network_trainer

SRC := $(wildcard $(SRCDIR)/*.c)
OBJ := $(patsubst $(SRCDIR)/%.c,$(SRCDIR)/%.o,$(SRC))

.PHONY: all run clean rebuild dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BINDIR)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -lm

run: all
	@echo "---- Running $(TARGET) ----"
	@$(TARGET)

clean:
	rm -f $(SRCDIR)/*.o $(TARGET)

rebuild: clean all
	@echo "Rebuilt."