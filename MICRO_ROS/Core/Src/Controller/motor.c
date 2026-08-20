#include "motor.h"
#include "microros_app.h"
#include "main.h"
#include "tim.h"
#include <stdint.h>
#include <math.h>

#define MAX_PWM_VALUE           65535
#define DELTA_TIME              10
#define PULSES_PER_REVOLUTION   1920.0
#define M_PI                    3.14159265358979323846f
#define DEADBAND_PWM            20000

#define SERVO_MIN_PWN           500
#define SERVO_MAX_PWM           2500

/* --- Motor Hardware Interface --- */

void Motor_Forward_L(uint16_t speed) {
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, 0);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, speed);
}

void Motor_Forward_R(uint16_t speed) {
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, 0);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, speed);
}

void Motor_Reverse_L(uint16_t speed) {
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, 0);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, speed);
}

void Motor_Reverse_R(uint16_t speed) {
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, 0);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, speed);
}

/* --- Global State Variables --- */

uint32_t previous_time = 0;

// Left Motor State
volatile float motor_rpm_L = 0.0;
volatile float target_rpm_L = 0.0;
volatile float position_L = 0.0;
volatile float current_pwm_L = 0.0;
uint32_t current_count_L;
uint32_t previous_count_L = 0;
float error_integral_L = 0.0;
float previous_error_L = 0.0;

// Right Motor State
volatile float motor_rpm_R = 0.0;
volatile float target_rpm_R = 0.0;
volatile float position_R = 0.0;
volatile float current_pwm_R = 0.0;
uint32_t current_count_R;
uint32_t previous_count_R = 0;
float error_integral_R = 0.0;
float previous_error_R = 0.0;

// Servo State
volatile float servo_pan_angle = 95;
volatile float servo_tilt_angle = 90;
volatile float previous_servo_pan_angle = 95;
volatile float previous_servo_tilt_angle = 90;

void Servo_Tilt_Rotate() {
    if (fabsf(servo_tilt_angle - previous_servo_tilt_angle) > 90.0) return;
    previous_servo_tilt_angle = servo_tilt_angle;

    float rad_angle = servo_tilt_angle*M_PI / 180.0;
    if (rad_angle < 0.0f) rad_angle = 0.0f;
    if (rad_angle > M_PI) rad_angle = M_PI;
    {
        float pwm = rad_angle*(2000.0/M_PI) + 500.0;
        __HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, (uint16_t)pwm);
    }
}
void Servo_Pan_Rotate() {
    if (fabsf(servo_pan_angle - previous_servo_pan_angle) > 90.0) return;
    previous_servo_pan_angle = servo_pan_angle;

    float rad_angle = servo_pan_angle*M_PI / 180.0;
    if (rad_angle < 0.0f) rad_angle = 0.0f;
    if (rad_angle > M_PI) rad_angle = M_PI;
    float pwm = rad_angle*(2000.0/M_PI) + 500.0;
    __HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, (uint16_t)pwm);
}

/* --- PID Gains --- */

volatile float Kf = 330.0;
volatile float Kp = 500.0;
volatile float Ki = 2000.0;
volatile float Kd = 0.0;

/* --- Interrupt Callback --- */

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    // System Tick handling
    if (htim->Instance == TIM7) {
        HAL_IncTick();
    }

    // Control Loop
    if (htim->Instance == TIM4) {
        // 1. Read Encoders
        current_count_L = __HAL_TIM_GET_COUNTER(&htim2);
        current_count_R = __HAL_TIM_GET_COUNTER(&htim3);

        int16_t delta_count_L = (int16_t)(current_count_L - previous_count_L);
        int16_t delta_count_R = (int16_t)(current_count_R - previous_count_R);

        previous_count_L = current_count_L;
        previous_count_R = current_count_R;

        // 2. Calculate Velocity and Position
        motor_rpm_L = ((float)delta_count_L / PULSES_PER_REVOLUTION) * (1000.0 / DELTA_TIME) * 60.0;
        motor_rpm_R = ((float)delta_count_R / PULSES_PER_REVOLUTION) * (1000.0 / DELTA_TIME) * 60.0;

        position_L += ((float)delta_count_L / PULSES_PER_REVOLUTION) * 2.0f * M_PI;
        position_R += ((float)delta_count_R / PULSES_PER_REVOLUTION) * 2.0f * M_PI;

        
        // ==========================================
        // LEFT MOTOR PID CONTROL
        // ==========================================
        float F_L = Kf * target_rpm_L;
        float error_L = target_rpm_L - motor_rpm_L;
        float P_L = Kp * error_L;

        error_integral_L += error_L * (DELTA_TIME / 1000.0);
        float I_L = Ki * error_integral_L;

        // Integral Clamping (Anti-Windup)
        if (I_L > 30000.0f) I_L = 30000.0f;
        else if (I_L < -30000.0f) I_L = -30000.0f;

        float derivative_L = (error_L - previous_error_L) / (DELTA_TIME / 1000.0);
        float D_L = Kd * derivative_L;
        previous_error_L = error_L;

        float pid_output_L = F_L + P_L + I_L + D_L;
        current_pwm_L = pid_output_L;

        // Output Saturation
        if (current_pwm_L > MAX_PWM_VALUE) current_pwm_L = MAX_PWM_VALUE;
        if (current_pwm_L < -MAX_PWM_VALUE) current_pwm_L = -MAX_PWM_VALUE;
      

        if (target_rpm_L == 0) {
            current_pwm_L = 0;
            error_integral_L = 0.0;
            previous_error_L = 0.0;
        }

        // else {
        //     if (current_pwm_L < DEADBAND_PWM && current_pwm_L > -DEADBAND_PWM) {
        //         current_pwm_L = 0.0;
        //     }
        // }
        // else {
        //     // Forward Deadband Handling
        //     if (current_pwm_L > 0) {
        //         if (current_pwm_L < DEADBAND_PWM) {
        //             current_pwm_L = (current_pwm_L > 0.0) ? DEADBAND_PWM : 0.0f;
        //         }
        //     } 
        //     // Reverse Deadband Handling
        //     else {
        //         if (current_pwm_L > -DEADBAND_PWM) {
        //             current_pwm_L = (current_pwm_L < 0.0) ? -DEADBAND_PWM : 0.0f;
        //         }
        //     }
        // }

        if (current_pwm_L >= 0.0) {
            Motor_Forward_L((uint16_t)current_pwm_L);
        } else {
            Motor_Reverse_L((uint16_t)(-current_pwm_L));
        }

        // ==========================================
        // RIGHT MOTOR PID CONTROL
        // ==========================================
        float F_R = Kf * target_rpm_R;
        float error_R = target_rpm_R - motor_rpm_R;
        float P_R = Kp * error_R;

        error_integral_R += error_R * (DELTA_TIME / 1000.0);
        float I_R = Ki * error_integral_R;

        // Integral Clamping (Anti-Windup)
        if (I_R > 30000.0f) I_R = 30000.0f;
        else if (I_R < -30000.0f) I_R = -30000.0f;

        float derivative_R = (error_R - previous_error_R) / (DELTA_TIME / 1000.0);
        float D_R = Kd * derivative_R;
        previous_error_R = error_R;

        float pid_output_R = F_R + P_R + I_R + D_R;
        current_pwm_R = pid_output_R;

        // Output Saturation
        if (current_pwm_R > MAX_PWM_VALUE) current_pwm_R = MAX_PWM_VALUE;
        if (current_pwm_R < -MAX_PWM_VALUE) current_pwm_R = -MAX_PWM_VALUE;

        if (target_rpm_R == 0) {
            current_pwm_R = 0;
            error_integral_R = 0.0;
            previous_error_R = 0.0;
        }

        // else {
        //     if (current_pwm_R < DEADBAND_PWM && current_pwm_R > -DEADBAND_PWM) {
        //         current_pwm_R = 0.0;
        //     }
        // }
        // else {
        //     // Forward Deadband Handling
        //     if (current_pwm_R > 0.0) {
        //         if (current_pwm_R < DEADBAND_PWM) {
        //             current_pwm_R = (current_pwm_R > 0.0) ? DEADBAND_PWM : 0.0f;
        //         }
        //     } 
        //     // Reverse Deadband Handling
        //     else {
        //         if (current_pwm_R > -DEADBAND_PWM) {
        //             current_pwm_R = (current_pwm_R < 0.0) ? -DEADBAND_PWM : 0.0f;
        //         }
        //     }
        // }

        if (current_pwm_R >= 0.0) {
            Motor_Forward_R((uint16_t)current_pwm_R);
        } else {
            Motor_Reverse_R((uint16_t)(-current_pwm_R));
        }

        // ==========================================
        // SERVO CONTROL
        // ==========================================

        Servo_Tilt_Rotate();
        Servo_Pan_Rotate();
           
    }
}