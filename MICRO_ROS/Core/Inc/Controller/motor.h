#ifndef MOTOR_H
#define MOTOR_H

#include <stdbool.h>
#include <stdint.h>

extern volatile float target_rpm_L;
extern volatile float target_rpm_R;
extern volatile float motor_rpm_L;
extern volatile float motor_rpm_R;
extern volatile float position_L;
extern volatile float position_R;
extern volatile float servo_pan_angle;
extern volatile float servo_tilt_angle;
void Servo_SetPanTarget(float angle);
void Servo_SetTiltTarget(float angle);
void Motor_SetWheelTargets(float left_rpm, float right_rpm);
bool Motor_WheelCommandIsFresh(uint32_t now);
extern volatile float Kf;
extern volatile float Kp;
extern volatile float Ki;
extern volatile float Kd;
extern volatile float current_pwm_L;
extern volatile float current_pwm_R;

#endif // MOTOR_H
