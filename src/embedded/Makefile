# Makefile pour SPEAR firmware Cortex-M4 / STM32F401RE
# Compilation: arm-none-eabi-gcc -O2 -mfloat-soft -mcpu=cortex-m4

# --- Compilation ---
CC = arm-none-eabi-gcc
CFLAGS = -O2 -g -Wall -Werror \
         -mfloat-soft -mcpu=cortex-m4 \
         -mlittle-endian -mthumb \
         -ffunction-sections -fdata-sections \
         -MD -std=c11

# --- Inclusion des chemins ---
INCLUDE_DIRS = -IC:\Users\Yuri\Documents\lightemulator\src\kernels \
               -IC:\Users\Yuri\Documents\lightemulator\src\embedded

# --- Définitions MISRA ---
MISRA_FLAGS = -DPROJECT_NAME=\"SPEAR_FIRMWARE\" \
              -DSTM32F401RE \
              -DCORTEX_M4

# --- Liens ---
LDFLAGS = -Tstm32f401re.ld \
          -Wl,--gc-sections \
          -mcpu=cortex-m4 -mfloat-soft -mthumb

# --- Bibliothèques ---
LIBS = -lc -lm -lnosys

# --- Sources ---
SRC = spear_firmware.c

# --- Objets ---
OBJ = $(SRC:.c=.o)

# --- Cible ---
TARGET = firmware.elf

# --- Règles par défaut ---
all: $(TARGET).bin

# --- Link ---
$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

# --- Compilation ----------
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) $(MISRA_FLAGS) -c -o $@ $<

# --- Binary ---
%.bin: $(TARGET)
	arm-none-eabi-objcopy -O binary -O ihex $< $@

# --- Nettoyage ---
clean:
	rm -f $(OBJ) $(TARGET) $(TARGET).bin $(TARGET).hex
	rm -f *.d

# --- Flash (optionnel) ---
# STM32F401RE Nucleo-64: flash at address 0x8000000
flash: $(TARGET).bin
	st-flash write $< 0x8000000

# --- Taille ---
size: $(TARGET)
	arm-none-eabi-size --format=raw --target=flash $<

# --- Vérification MISRA (optionnel) ---
# requires: misra-c-checker
misra:
	@echo "MISRA checking not implemented - use tools separately"

# --- Aide ---
help:
	@echo "Cible:"
	@echo "  make        - Compile firmware.elf"
	@echo "  make bin    - Generate firmware.bin"
	@echo "  make flash  - Flash Nucleo-64"
	@echo "  make size   - Show binary size"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - This help"