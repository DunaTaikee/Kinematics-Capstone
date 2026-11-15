# 16-384 Capstone

Capstone project for 16-384 Robot Kinematics and Dynamics, Fall 2026.

## Team

- Tait
- Vern

## Milestone 1

### Kinematics
Owner: Vern

- Forward kinematics
- 6x7 Jacobian
- FK reference tests
- Jacobian finite-difference validation

### Inverse Differential Kinematics and Demos
Owner: Tait

- Inverse differential kinematics
- Cartesian motion controller
- +x, +y, +z motion demo
- z-axis rotation demo
- Lightsaber grasp demo

## Function Interfaces

### `forward_kinematics(q)`

Input:
- `q`: joint angles, shape `(7,)`

Returns:
- end-effector homogeneous transform, shape `(4, 4)`

### `jacobian(q)`

Input:
- `q`: joint angles, shape `(7,)`

Returns:
- Jacobian, shape `(6, 7)`

Row convention:

```text
[vx, vy, vz, wx, wy, wz]