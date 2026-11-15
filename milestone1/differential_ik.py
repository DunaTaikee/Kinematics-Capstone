import time
import numpy as np

from milestone1.kinematics import jacobian


def differential_ik_step(q, desired_twist):
    """
    Compute the joint velocity needed for a desired end-effector twist.

    desired_twist:
        [vx, vy, vz, wx, wy, wz]
    """
    q = np.asarray(q, dtype=float)
    desired_twist = np.asarray(desired_twist, dtype=float)

    J = jacobian(q)

    # q_dot = J^+ x_dot
    q_dot = np.linalg.pinv(J) @ desired_twist

    return q_dot


def move_cartesian(robot, displacement, duration=4.0, rate=100):
    """
    Move the end effector by a Cartesian displacement using
    inverse differential kinematics.

    displacement:
        [dx, dy, dz, rx, ry, rz]

    Translation is in meters.
    Rotation is in radians about the base-frame axes.
    """
    displacement = np.asarray(displacement, dtype=float)

    if displacement.shape != (6,):
        raise ValueError("displacement must have shape (6,)")

    dt = 1.0 / rate
    steps = int(duration * rate)

    # Constant desired Cartesian velocity
    desired_twist = displacement / duration

    # Start the commanded trajectory from the robot's actual position
    q_command = np.asarray(robot.joint_values, dtype=float).copy()

    for _ in range(steps):

        # Use measured q to recompute the Jacobian each step
        q_measured = np.asarray(robot.joint_values, dtype=float)

        q_dot = differential_ik_step(
            q_measured,
            desired_twist,
        )

        # Integrate the commanded trajectory continuously
        q_command = q_command + q_dot * dt

        # Check before sending anything to the robot
        violation = robot.check_safety(q_command)

        if violation is not None:
            robot.stop()
            raise RuntimeError(
                f"Cartesian motion would violate safety: {violation}"
            )

        # Send both position and known joint velocity
        robot.servo_joints(
            q_command,
            velocities=q_dot,
        )

        time.sleep(dt)

    robot.stop()