#include "imu.h"
#include "microros_app.h"
#include "main.h"
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// ===== MPU 6050 CONFIGURATIONS =====
#define MPU6050_I2C_ADDR         0xD0 // 0x68 shifted to the left by 1

// Registers
#define MPU6050_SMPLRT_DIV       0x19 // Register 25
#define MPU6050_CONFIG           0x1A // Register 26
#define MPU6050_GYRO_CONFIG      0x1B // Register 27
#define MPU6050_ACCEL_CONFIG     0x1C // Register 28
#define MPU6050_PWR_MGMT_1       0x6B // Register 107
#define MPU6050_WHO_AM_I         0x75 // Register 117

#define MPU6050_ACCEL_XOUT_H     0x3B // Register 59
#define MPU6050_GYRO_XOUT_H      0x43 // Register 67

// The expected WHO_AM_I value for MPU6050 is typically 0x68
#define MPU6050_WHO_AM_I_VAL     0x68
#define MPU_I2C_TIMEOUT          20  // Reduced from 100ms to 20ms

// Variables
int16_t Accel_X_RAW = 0, Accel_Y_RAW = 0, Accel_Z_RAW = 0;
int16_t Gyro_X_RAW = 0, Gyro_Y_RAW = 0, Gyro_Z_RAW = 0;

volatile float Ax, Ay, Az;
volatile float Gx, Gy, Gz;
volatile float roll = 0, pitch = 0, yaw = 0;

float Ax_bias = 0, Ay_bias = 0, Az_bias = 0;
float Gx_bias = 0, Gy_bias = 0, Gz_bias = 0;
#define MPU_CALIBRATION_SAMPLES  500

#define COMPLEMENTARY_FILTER_ALPHA  0.98f
uint32_t prev_tick = 0;

bool mpu_calibration_done = false;
bool mpu_init_status = false;

// Functions
bool MPU6050_Init(I2C_HandleTypeDef *hi2c) {

    uint8_t check = 0;
    uint8_t data;

    HAL_StatusTypeDef status;
    status = HAL_I2C_Mem_Read(hi2c, MPU6050_I2C_ADDR, MPU6050_WHO_AM_I, I2C_MEMADD_SIZE_8BIT, &check, 1, MPU_I2C_TIMEOUT);

    // Check WHO_AM_I register to confirm device identity
    if (status != HAL_OK) {
        return false;
    }

    if (check != MPU6050_WHO_AM_I_VAL) {
        return false;
    }

    // Wake up the device and set clock source (Auto select best available clock)
    data = 0x01;
    if (HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_PWR_MGMT_1, I2C_MEMADD_SIZE_8BIT, &data, 1, MPU_I2C_TIMEOUT) != HAL_OK) {
        return false;
    }
    HAL_Delay(100);

    // Set Data Rate (Sample Rate Divider)
    data = 0x00; // 1kHz sample rate
    HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_SMPLRT_DIV, I2C_MEMADD_SIZE_8BIT, &data, 1, MPU_I2C_TIMEOUT);

    // Set Configuration (Digital Low Pass Filter for BOTH Gyro and Accel in MPU6050)
    data = 0x03; // ~42Hz bandwidth
    HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_CONFIG, I2C_MEMADD_SIZE_8BIT, &data, 1, MPU_I2C_TIMEOUT);

    // Configure Gyroscope
    data = 0x08; // +/- 500 dps
    HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_GYRO_CONFIG, I2C_MEMADD_SIZE_8BIT, &data, 1, MPU_I2C_TIMEOUT);

    // Configure Accelerometer
    data = 0x08; // +/- 4g
    HAL_I2C_Mem_Write(hi2c, MPU6050_I2C_ADDR, MPU6050_ACCEL_CONFIG, I2C_MEMADD_SIZE_8BIT, &data, 1, MPU_I2C_TIMEOUT);

    mpu_init_status = true;
    return true;
}

bool MPU6050_Read_Accel(I2C_HandleTypeDef *hi2c) {
    uint8_t Rec_Data[6];

    if (HAL_I2C_Mem_Read(hi2c, MPU6050_I2C_ADDR, MPU6050_ACCEL_XOUT_H, I2C_MEMADD_SIZE_8BIT, Rec_Data, 6, MPU_I2C_TIMEOUT) == HAL_OK) {
        Accel_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
        Accel_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
        Accel_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

        // With +/- 4g, the sensitivity factor is 8192 LSB/g.
        Ax = Accel_X_RAW / 8192.0;
        Ay = Accel_Y_RAW / 8192.0;
        Az = Accel_Z_RAW / 8192.0;

        if (mpu_calibration_done) {
            Ax -= Ax_bias;
            Ay -= Ay_bias;
            Az -= Az_bias;
        }
        return true;
    }
    return false;
}

bool MPU6050_Read_Gyro(I2C_HandleTypeDef *hi2c) {
    uint8_t Rec_Data[6];

    if (HAL_I2C_Mem_Read(hi2c, MPU6050_I2C_ADDR, MPU6050_GYRO_XOUT_H, I2C_MEMADD_SIZE_8BIT, Rec_Data, 6, MPU_I2C_TIMEOUT) == HAL_OK) {
        Gyro_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
        Gyro_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
        Gyro_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

        // With +/- 500 dps, the sensitivity factor is 65.5 LSB/dps.
        Gx = Gyro_X_RAW / 65.5;
        Gy = Gyro_Y_RAW / 65.5;
        Gz = Gyro_Z_RAW / 65.5;

        if (mpu_calibration_done) {
            Gx -= Gx_bias;
            Gy -= Gy_bias;
            Gz -= Gz_bias;
        }
        return true;
    }
    return false;
}

// Optimized: Read both accel and gyro in one burst (14 bytes total)
// MPU6050 registers are sequential: ACCEL (0x3B-0x40) + TEMP (0x41-0x42) + GYRO (0x43-0x48)
bool MPU6050_Read_All(I2C_HandleTypeDef *hi2c) {
    uint8_t Rec_Data[14];

    // Read all 14 bytes in one I2C transaction (accel + temp + gyro)
    if (HAL_I2C_Mem_Read(hi2c, MPU6050_I2C_ADDR, MPU6050_ACCEL_XOUT_H, I2C_MEMADD_SIZE_8BIT, Rec_Data, 14, MPU_I2C_TIMEOUT) == HAL_OK) {
        // Parse accelerometer (bytes 0-5)
        Accel_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
        Accel_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
        Accel_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

        Ax = Accel_X_RAW / 8192.0;
        Ay = Accel_Y_RAW / 8192.0;
        Az = Accel_Z_RAW / 8192.0;

        // Parse gyroscope (bytes 8-13, skip temp at 6-7)
        Gyro_X_RAW = (int16_t)(Rec_Data[8] << 8 | Rec_Data[9]);
        Gyro_Y_RAW = (int16_t)(Rec_Data[10] << 8 | Rec_Data[11]);
        Gyro_Z_RAW = (int16_t)(Rec_Data[12] << 8 | Rec_Data[13]);

        Gx = Gyro_X_RAW / 65.5;
        Gy = Gyro_Y_RAW / 65.5;
        Gz = Gyro_Z_RAW / 65.5;

        // Apply calibration
        if (mpu_calibration_done) {
            Ax -= Ax_bias;
            Ay -= Ay_bias;
            Az -= Az_bias;
            Gx -= Gx_bias;
            Gy -= Gy_bias;
            Gz -= Gz_bias;
        }
        return true;
    }
    return false;
}

void MPU6050_Calibrate(I2C_HandleTypeDef *hi2c) {
    float ax_sum = 0, ay_sum = 0, az_sum = 0;
    float gx_sum = 0, gy_sum = 0, gz_sum = 0;

    for (int i = 0; i < MPU_CALIBRATION_SAMPLES; i++) {
        MPU6050_Read_Accel(hi2c);
        MPU6050_Read_Gyro(hi2c);

        ax_sum += Ax;
        ay_sum += Ay;
        az_sum += Az;
        gx_sum += Gx;
        gy_sum += Gy;
        gz_sum += Gz;

        HAL_Delay(5);
    }
    Ax_bias = ax_sum / MPU_CALIBRATION_SAMPLES;
    Ay_bias = ay_sum / MPU_CALIBRATION_SAMPLES;
    Az_bias = (az_sum / MPU_CALIBRATION_SAMPLES) - 1.0f;
    Gx_bias = gx_sum / MPU_CALIBRATION_SAMPLES;
    Gy_bias = gy_sum / MPU_CALIBRATION_SAMPLES;
    Gz_bias = gz_sum / MPU_CALIBRATION_SAMPLES;

    mpu_calibration_done = true;
}

void Calculate_Angles(void) {
    uint32_t current_tick = HAL_GetTick();
    float dt;

    if (prev_tick == 0) {
        dt = 0.01f;
    } else {
        dt = (current_tick - prev_tick) / 1000.0f;
    }
    prev_tick = current_tick;

    if (dt > 0.1f) dt = 0.1f;

    roll += Gx * dt;
    pitch += Gy * dt;
    yaw += Gz * dt;

    float accel_roll = atan2(Ay, Az) * (180.0f / M_PI);
    float accel_pitch = atan2(-Ax, sqrt(Ay*Ay + Az*Az)) * (180.0f / M_PI);

    roll = COMPLEMENTARY_FILTER_ALPHA * roll + (1.0f - COMPLEMENTARY_FILTER_ALPHA) * accel_roll;
    pitch = COMPLEMENTARY_FILTER_ALPHA * pitch + (1.0f - COMPLEMENTARY_FILTER_ALPHA) * accel_pitch;
}

uint32_t last_reconnect_attempt = 0;
const uint32_t RECONNECT_INTERVAL = 2000;

bool Read_IMU(void) {
    if (mpu_init_status == true) {
        // Use optimized combined read (1 I2C transaction instead of 2)
        bool read_ok = MPU6050_Read_All(&hi2c1);

        if (!read_ok) {
          mpu_init_status = false;
          debug_print("[ERROR] IMU I2C Read Failed! Bus offline.");
          return false;
        }
        else {
            return true;
        }
  }
  else {
    if (HAL_GetTick() - last_reconnect_attempt >= RECONNECT_INTERVAL) {
      last_reconnect_attempt = HAL_GetTick();
      debug_print("[WARN] Attempting IMU I2C bus recovery...");

      HAL_I2C_DeInit(&hi2c1);
      I2C1_Clear_Busy_Flag();
      __HAL_RCC_I2C1_FORCE_RESET();
      osDelay(2);
      __HAL_RCC_I2C1_RELEASE_RESET();
      MX_I2C1_Init(); 
      
      mpu_init_status = MPU6050_Init(&hi2c1);
      
      if (mpu_init_status) {
          debug_print("[SUCCESS] IMU Reconnected!");
      } else {
          debug_print("[FAILED] IMU Reconnect failed, retrying in 2s...");
      }
    }
    return false;
  }
}