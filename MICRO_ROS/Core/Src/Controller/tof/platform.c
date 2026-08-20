#include "platform.h"

// Pulls in the I2C handle configured in main.c (Make sure it's hi2c1)
extern I2C_HandleTypeDef hi2c1; 

uint8_t VL53L7CX_RdByte(
		VL53L7CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_value)
{
	// HAL_I2C_Mem_Read automatically handles the 16-bit register address
	uint8_t status = HAL_I2C_Mem_Read(&hi2c1, p_platform->address, RegisterAdress, I2C_MEMADD_SIZE_16BIT, p_value, 1, 100);
	return status;
}

uint8_t VL53L7CX_WrByte(
		VL53L7CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t value)
{
	uint8_t status = HAL_I2C_Mem_Write(&hi2c1, p_platform->address, RegisterAdress, I2C_MEMADD_SIZE_16BIT, &value, 1, 100);
	return status;
}

uint8_t VL53L7CX_WrMulti(
		VL53L7CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_values,
		uint32_t size)
{
    // Firmware loading can take a while, so the timeout is set higher (1000ms)
	uint8_t status = HAL_I2C_Mem_Write(&hi2c1, p_platform->address, RegisterAdress, I2C_MEMADD_SIZE_16BIT, p_values, size, 1000);
	return status;
}

uint8_t VL53L7CX_RdMulti(
		VL53L7CX_Platform *p_platform,
		uint16_t RegisterAdress,
		uint8_t *p_values,
		uint32_t size)
{
	uint8_t status = HAL_I2C_Mem_Read(&hi2c1, p_platform->address, RegisterAdress, I2C_MEMADD_SIZE_16BIT, p_values, size, 1000);
	return status;
}

uint8_t VL53L7CX_Reset_Sensor(VL53L7CX_Platform *p_platform)
{
	/* (Optional) Need to be implemented by customer. This function returns 0 if OK */
	/* We handle the reset sequence (LPn pin) in main.c during initialization, 
       so this can be left blank safely. */
	return 0;
}

void VL53L7CX_SwapBuffer(
		uint8_t 		*buffer,
		uint16_t 	 	 size)
{
	uint32_t i, tmp;

	/* Example of possible implementation using <string.h> */
	for(i = 0; i < size; i = i + 4)
	{
		tmp = (buffer[i]<<24) | (buffer[i+1]<<16) | (buffer[i+2]<<8) | (buffer[i+3]);
		memcpy(&(buffer[i]), &tmp, 4);
	}
}

uint8_t VL53L7CX_WaitMs(
		VL53L7CX_Platform *p_platform,
		uint32_t TimeMs)
{
	HAL_Delay(TimeMs);
	return 0;
}