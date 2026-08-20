#ifndef MOTOR_H
#define MOTOR_H

extern volatile float target_rpm_L;
extern volatile float target_rpm_R;
extern volatile float motor_rpm_L;
extern volatile float motor_rpm_R;
extern volatile float position_L;
extern volatile float position_R;
extern volatile float servo_pan_angle;
extern volatile float servo_tilt_angle;
extern volatile float Kf;
extern volatile float Kp;
extern volatile float Ki;
extern volatile float Kd;
extern volatile float current_pwm_L;
extern volatile float current_pwm_R;

#endif // MOTOR_H