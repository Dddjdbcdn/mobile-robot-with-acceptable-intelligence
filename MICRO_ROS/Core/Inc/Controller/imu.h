#ifndef IMU_H
#define IMU_H

#include "i2c.h"
#include <stdbool.h>

extern volatile float Ax;
extern volatile float Ay;
extern volatile float Az;
extern volatile float Gx;
extern volatile float Gy;
extern volatile float Gz;
extern volatile bool mpu_init_status;
extern bool mpu_calibration_done;
extern volatile uint32_t imu_last_read_ms;
extern volatile uint32_t imu_read_error_count;
extern volatile uint32_t imu_reconnect_count;
extern volatile uint8_t imu_consecutive_read_errors;

bool MPU6050_Init(I2C_HandleTypeDef *hi2c);
bool MPU6050_Read_Accel(I2C_HandleTypeDef *hi2c);
bool MPU6050_Read_Gyro(I2C_HandleTypeDef *hi2c);
bool MPU6050_Calibrate(I2C_HandleTypeDef *hi2c);
bool Read_IMU(void);

#endif // IMU_H
