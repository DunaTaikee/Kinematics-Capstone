from xarm7_lib import Robot

with Robot() as robot:
    robot.free_drive()
    input('Enter to stop\n')
    robot.end_free_drive()