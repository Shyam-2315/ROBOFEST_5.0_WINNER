from pymavlink import mavutil
import time

master = mavutil.mavlink_connection('COM5', baud=115200)

master.wait_heartbeat()
print("Connected to Pixhawk")

def set_servo(channel, pwm):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel,
        pwm,
        0,0,0,0,0
    )

while True:
    set_servo(1, 1000)
    set_servo(2, 1000)
    time.sleep(2)

    set_servo(1, 1500)
    set_servo(2, 1500)
    time.sleep(2)

    set_servo(1, 2000)
    set_servo(2, 2000)
    time.sleep(2)