#ifndef IMU_H
#define IMU_H

#include "i2c.h"
#include <stdbool.h>

extern volatile float Ax;
extern volatile float Ay;
extern volatile float Gz;

bool MPU6050_Init(I2C_HandleTypeDef *hi2c);
bool MPU6050_Read_Accel(I2C_HandleTypeDef *hi2c);
bool MPU6050_Read_Gyro(I2C_HandleTypeDef *hi2c);
void MPU6050_Calibrate(I2C_HandleTypeDef *hi2c);
bool Read_IMU(void);

#endif // IMU_H