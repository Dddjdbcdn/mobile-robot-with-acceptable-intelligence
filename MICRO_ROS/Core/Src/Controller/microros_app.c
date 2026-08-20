#include "main.h"
#include "cmsis_os.h"
#include "dma.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

#include "microros_app.h"
#include "imu.h"
#include "motor.h"
#include "ultrasonic.h"
#include "vl53l7cx_api.h"
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <uxr/client/transport.h>
#include <rmw_microxrcedds_c/config.h>
#include <rmw_microros/rmw_microros.h>

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <sensor_msgs/msg/joint_state.h>
#include <std_msgs/msg/string.h>
#include <std_msgs/msg/float32.h>
#include <geometry_msgs/msg/point32.h>
#include <std_msgs/msg/int16_multi_array.h>

extern void I2C1_Clear_Busy_Flag(void);
extern void MX_I2C1_Init(void);

extern VL53L7CX_ResultsData Results;
extern uint8_t tof_alive_status;
extern uint8_t tof_data_ready;

#define PI 3.14159265358979323846f
#define WHEEL_RADIUS 0.0325
#define WHEEL_SEPERATION 0.396

// --- TRANSPORT DECLARATIONS ---
bool cubemx_transport_open(struct uxrCustomTransport * transport);
bool cubemx_transport_close(struct uxrCustomTransport * transport);
size_t cubemx_transport_write(struct uxrCustomTransport* transport, const uint8_t * buf, size_t len, uint8_t * err);
size_t cubemx_transport_read(struct uxrCustomTransport* transport, uint8_t* buf, size_t len, int timeout, uint8_t* err);

void * microros_allocate(size_t size, void * state);
void microros_deallocate(void * pointer, void * state);
void * microros_reallocate(void * pointer, size_t size, void * state);
void * microros_zero_allocate(size_t number_of_elements, size_t size_of_element, void * state);

// ==========================================
// GLOBAL MICRO-ROS VARIABLES 
// ==========================================
rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rclc_executor_t executor;

// --- PUBLISHERS ---
rcl_publisher_t debug_publisher;
std_msgs__msg__String debug_msg;

rcl_publisher_t imu_publisher;
geometry_msgs__msg__Point32 imu_msg;

rcl_publisher_t ultrasonic_publisher;
geometry_msgs__msg__Point32 ultrasonic_msg;

rcl_publisher_t wheel_state_publisher;
sensor_msgs__msg__JointState wheel_state_msg;

rcl_publisher_t pwm_publisher;
geometry_msgs__msg__Point32 pwm_msg;

rcl_publisher_t tof_raw_publisher;
std_msgs__msg__Int16MultiArray tof_raw_msg;

// Publisher Memory Buffers
rosidl_runtime_c__String pub_names[2];
double pub_positions[2];
double pub_velocities[2];
char pub_name_L[] = "left_wheel_joint";
char pub_name_R[] = "right_wheel_joint";

int16_t allocated_data_buffer[64];

// --- SUBSCRIBER ---
rcl_subscription_t wheel_cmd_subscriber;
sensor_msgs__msg__JointState wheel_cmd_msg;

// PWM Command Subscribers
rcl_subscription_t servo_tilt_subscriber;
std_msgs__msg__Float32 servo_tilt_msg;

rcl_subscription_t servo_pan_subscriber;
std_msgs__msg__Float32 servo_pan_msg;

// Subscriber Memory Buffers
rosidl_runtime_c__String sub_names[2];
char sub_name_string_L[30];
char sub_name_string_R[30];
double sub_positions[2];
double sub_velocities[2];
double sub_efforts[2];

// --- TIMERS ---
// --- Initialize Timers ---
rcl_timer_t wheel_timer;
rcl_timer_t imu_timer;
rcl_timer_t ultrasonic_timer;
rcl_timer_t tof_timer;
rcl_timer_t pwm_timer;

// ==========================================
// CALLBACKS & FUNCTIONS
// ==========================================

void servo_tilt_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    servo_tilt_angle = msg->data;
}

void servo_pan_callback(const void * msgin)
{
    const std_msgs__msg__Float32 * msg = (const std_msgs__msg__Float32 *)msgin;
    servo_pan_angle = msg->data;
}

uint8_t count = 0;
void debug_print(const char *format, ...)
{
        static char buffer[128];

          va_list args;
          va_start(args, format);
          vsnprintf(buffer, sizeof(buffer), format, args);
          va_end(args);

          debug_msg.data.data = buffer;
          debug_msg.data.size = strlen(buffer);
          debug_msg.data.capacity = sizeof(buffer);

          rcl_ret_t ret;

          ret=rcl_publish(&debug_publisher, &debug_msg, NULL);
          (void)ret;

}

void wheel_cmd_callback(const void * msgin)
{
    const sensor_msgs__msg__JointState * msg = (const sensor_msgs__msg__JointState *)msgin;

    if (msg->velocity.size >= 2)
    {
        float target_rad_s_L = (float)msg->velocity.data[0];
        float target_rad_s_R = (float)msg->velocity.data[1];

        target_rpm_L = target_rad_s_L * (60.0f / (2.0f * PI));
        target_rpm_R = target_rad_s_R * (60.0f / (2.0f * PI));
    }
}

void wheel_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      wheel_state_msg.position.data[0] = (position_L == 0.0f) ? 1e-6 : (double)position_L;
      wheel_state_msg.position.data[1] = (position_R == 0.0f) ? 1e-6 : (double)position_R;
      wheel_state_msg.velocity.data[0] = (double)(motor_rpm_L * 2.0f * PI / 60.0f);
      wheel_state_msg.velocity.data[1] = (double)(motor_rpm_R * 2.0f * PI / 60.0f);

      rcl_ret_t ret = rcl_publish(&wheel_state_publisher, &wheel_state_msg, NULL);
      (void)ret;
    }
}

void imu_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      imu_msg.x = Ax;
      imu_msg.y = Ay;
      imu_msg.z = Gz;
      rcl_ret_t ret = rcl_publish(&imu_publisher, &imu_msg, NULL);
      (void)ret;
    }
}

void ultrasonic_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      ultrasonic_msg.x = final_dist[0] / 1000.0f;
      ultrasonic_msg.y = final_dist[1] / 1000.0f;
      ultrasonic_msg.z = final_dist[2] / 1000.0f;
      rcl_ret_t ret = rcl_publish(&ultrasonic_publisher, &ultrasonic_msg, NULL);
      (void)ret;
    }
}

void tof_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer == NULL) return;

    vTaskSuspendAll();

    for (uint8_t i = 0; i < 64; i++)
    {
      tof_raw_msg.data.data[i] = Results.distance_mm[i];
        uint8_t status = Results.target_status[i];

        if (status == 5) {
            tof_raw_msg.data.data[i] = Results.distance_mm[i];
        } else {
            tof_raw_msg.data.data[i] = -1; 
        }
    }
    
    xTaskResumeAll(); 

    rcl_ret_t ret = rcl_publish(&tof_raw_publisher, &tof_raw_msg, NULL);
    (void)ret;
}

void pwm_timer_callback(rcl_timer_t * timer, int64_t last_call_time)
{
    if (timer != NULL) {
      pwm_msg.x = current_pwm_L;
      pwm_msg.y = current_pwm_R;
      pwm_msg.z = 0.0;
      rcl_ret_t ret = rcl_publish(&pwm_publisher, &pwm_msg, NULL);
      (void)ret;
    }
}

// ==========================================
// MAIN TASK
// ==========================================
void run_microros_app()
{
    // --- 2. Micro-ROS Transport Setup ---
    rmw_uros_set_custom_transport(
      true,
      (void *) &huart1,
      cubemx_transport_open,
      cubemx_transport_close,
      cubemx_transport_write,
      cubemx_transport_read);

    rcl_allocator_t freeRTOS_allocator = rcutils_get_zero_initialized_allocator();
    freeRTOS_allocator.allocate = microros_allocate;
    freeRTOS_allocator.deallocate = microros_deallocate;
    freeRTOS_allocator.reallocate = microros_reallocate;
    freeRTOS_allocator.zero_allocate =  microros_zero_allocate;

    if (!rcutils_set_default_allocator(&freeRTOS_allocator)) {
        printf("Error on default allocators (line %d)\n", __LINE__);
    }

    allocator = rcl_get_default_allocator();

    rclc_support_init(&support, 0, NULL, &allocator);
    rclc_node_init_default(&node, "cubemx_node", "", &support);

    // --- 3. WIRING MEMORY ---
    // Publisher Memory

    // WHEEL
    wheel_state_msg.name.data = pub_names;
    wheel_state_msg.name.size = 2;
    wheel_state_msg.name.capacity = 2;

    wheel_state_msg.name.data[0].data = pub_name_L;
    wheel_state_msg.name.data[0].size = strlen(pub_name_L);
    wheel_state_msg.name.data[0].capacity = strlen(pub_name_L) + 1;

    wheel_state_msg.name.data[1].data = pub_name_R;
    wheel_state_msg.name.data[1].size = strlen(pub_name_R);
    wheel_state_msg.name.data[1].capacity = strlen(pub_name_R) + 1;

    wheel_state_msg.position.data = pub_positions;
    wheel_state_msg.position.size = 2;
    wheel_state_msg.position.capacity = 2;

    wheel_state_msg.velocity.data = pub_velocities;
    wheel_state_msg.velocity.size = 2;
    wheel_state_msg.velocity.capacity = 2;

    // TOF
    // Map the internal pointer to our static memory block
    tof_raw_msg.data.data = allocated_data_buffer;
    tof_raw_msg.data.size = 64;
    tof_raw_msg.data.capacity = 64;

    // Configure the multi-array layout parameters to 0 since it's treated as a flat stream
    tof_raw_msg.layout.dim.size = 0;
    tof_raw_msg.layout.dim.capacity = 0;
    tof_raw_msg.layout.dim.data = NULL;
    tof_raw_msg.layout.data_offset = 0;

    // Subscriber Memory
    wheel_cmd_msg.name.data = sub_names;
    wheel_cmd_msg.name.capacity = 2;
    wheel_cmd_msg.name.size = 0;

    wheel_cmd_msg.name.data[0].data = sub_name_string_L;
    wheel_cmd_msg.name.data[0].capacity = 30;
    wheel_cmd_msg.name.data[0].size = 0;

    wheel_cmd_msg.name.data[1].data = sub_name_string_R;
    wheel_cmd_msg.name.data[1].capacity = 30;
    wheel_cmd_msg.name.data[1].size = 0;

    wheel_cmd_msg.velocity.data = sub_velocities;
    wheel_cmd_msg.velocity.capacity = 2;
    wheel_cmd_msg.velocity.size = 0;

    wheel_cmd_msg.position.data = sub_positions;
    wheel_cmd_msg.position.capacity = 2;
    wheel_cmd_msg.position.size = 0;

    wheel_cmd_msg.effort.data = sub_efforts;
    wheel_cmd_msg.effort.capacity = 2;
    wheel_cmd_msg.effort.size = 0;

    // --- 4. Initialize ROS 2 Entities ---
    rclc_publisher_init_best_effort(
      &imu_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/imu_msg");

    rclc_publisher_init_best_effort(
      &ultrasonic_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/ultrasonic_msg");

    rclc_publisher_init_best_effort(
      &tof_raw_publisher, &node, 
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int16MultiArray), "stm32/tof_raw_data"); //Int16MultiArray

    rclc_publisher_init_best_effort(
      &debug_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, String), "stm32/debug_msg");

    rclc_publisher_init_best_effort(
      &pwm_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Point32), "stm32/pwm_msg");

    rclc_publisher_init_best_effort(
      &wheel_state_publisher, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
      "stm32/wheel_states");

    rclc_subscription_init_default(
      &wheel_cmd_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
      "stm32/wheel_commands");

    rclc_subscription_init_default(
      &servo_tilt_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "stm32/servo_tilt");

    rclc_subscription_init_default(
      &servo_pan_subscriber, &node,
      ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32),
      "stm32/servo_pan");

    rclc_timer_init_default2(
        &wheel_timer, &support, RCL_MS_TO_NS(20), wheel_timer_callback, true); // 50Hz
    rclc_timer_init_default2(
        &imu_timer, &support, RCL_MS_TO_NS(20), imu_timer_callback, true); // 50Hz
    rclc_timer_init_default2(
        &ultrasonic_timer, &support, RCL_MS_TO_NS(50), ultrasonic_timer_callback, true); // 20Hz
    rclc_timer_init_default2(
        &tof_timer, &support, RCL_MS_TO_NS(100), tof_timer_callback, true); // 10Hz
    rclc_timer_init_default2(
        &pwm_timer, &support, RCL_MS_TO_NS(50), pwm_timer_callback, true); // 20Hz

    rclc_executor_init(&executor, &support.context, 10, &allocator);
    rclc_executor_add_subscription(&executor, &wheel_cmd_subscriber, &wheel_cmd_msg, &wheel_cmd_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &servo_tilt_subscriber, &servo_tilt_msg, &servo_tilt_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &servo_pan_subscriber, &servo_pan_msg, &servo_pan_callback, ON_NEW_DATA);
    rclc_executor_add_timer(&executor, &wheel_timer);
    rclc_executor_add_timer(&executor, &imu_timer);
    rclc_executor_add_timer(&executor, &ultrasonic_timer);
    rclc_executor_add_timer(&executor, &tof_timer);
    rclc_executor_add_timer(&executor, &pwm_timer);

    for(;;)
    {
        rclc_executor_spin_some(&executor, RCL_MS_TO_NS(100));

        osDelay(1);  
    }
}

