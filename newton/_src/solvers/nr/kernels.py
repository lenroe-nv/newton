# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warp as wp
from ...sim import (
    Contacts,
    Control,
    JointMode,
    JointType,
    Model,
    State,
)

from ..solver import integrate_bodies

@wp.func
def solve6x6_gaussian(A: wp.spatial_matrixf, b: wp.spatial_vectorf) -> wp.spatial_vectorf:
    """
    Solve the 6x6 linear system Ax = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: 6x6 spatial matrix (coefficients)
        b: 6x1 spatial vector (right-hand side)
        
    Returns:
        x: 6x1 spatial vector (solution)
    """
    # Create working copies
    # Note: spatial matrices and vectors are passed by value, so we can modify them directly
    
    # Forward elimination with partial pivoting
    for k in range(5):  # 0 to 4 (n-1)
        # Find pivot (largest absolute value in column k, from row k onwards)
        max_val = float(wp.abs(A[k, k]))  # Declare as dynamic variable
        pivot_row = int(k)  # Declare as dynamic variable
        
        for i in range(k + 1, 6):
            if wp.abs(A[i, k]) > max_val:
                max_val = wp.abs(A[i, k])
                pivot_row = i
        
        # Swap rows if needed
        if pivot_row != k:
            # Swap rows in matrix A
            for j in range(6):
                temp = A[k, j]
                A[k, j] = A[pivot_row, j]
                A[pivot_row, j] = temp
            
            # Swap elements in vector b
            temp_b = b[k]
            b[k] = b[pivot_row]
            b[pivot_row] = temp_b
        
        # Check for zero pivot (singular matrix)
        if wp.abs(A[k, k]) < 1e-12:
            # Return zero vector for singular systems
            return wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Eliminate column k in rows k+1 to 5
        for i in range(k + 1, 6):
            factor = float(A[i, k] / A[k, k])  # Declare as dynamic variable
            
            # Update row i
            for j in range(k, 6):
                A[i, j] = A[i, j] - factor * A[k, j]
            
            b[i] = b[i] - factor * b[k]
    
    # Back substitution
    x = wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    for i in range(5, -1, -1):  # 5 down to 0
        sum_val = float(0.0)  # Declare as dynamic variable
        for j in range(i + 1, 6):
            sum_val += A[i, j] * x[j]
        
        x[i] = (b[i] - sum_val) / A[i, i]
    
    return x


@wp.kernel
def add_gravity_kernel(
    body_mass: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    
    mass = body_mass[tid]

    gravity_force = gravity[0] * mass
    current_f = body_f[tid]
    new_f = wp.spatial_vector(
        wp.spatial_top(current_f) + gravity_force,
        wp.spatial_bottom(current_f)
    )
    body_f[tid] = new_f


@wp.kernel  
def add_external_wrenches_kernel(
    external_wrenches: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    external_wrench = external_wrenches[tid]
    current_f = body_f[tid]
    body_f[tid] = current_f + external_wrench


@wp.kernel
def compute_residual_and_jacobian(
    body_q_old: wp.array(dtype=wp.transform),
    body_qd_old: wp.array(dtype=wp.spatial_vector),
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    contact_f: wp.array(dtype=wp.spatial_vector),
    contact_jac_pos: wp.array(dtype=wp.spatial_matrixf),  # Contact jacobian w.r.t. positions
    contact_jac_vel: wp.array(dtype=wp.spatial_matrixf),  # Contact jacobian w.r.t. velocities
    dt: float,
    angular_damping: float,
    residual: wp.array(dtype=wp.spatial_vectorf),
    jacobian: wp.array(dtype=wp.spatial_matrixf),
):
    """
    Compute the residual and jacobian for the implicit system.
    
    r = M(qd_new - qd_old) - dt * f(q_new, qd_new)
    J = M - dt (∂f/∂qd + ∂f/∂q * ∂q/∂qd)
    """
    tid = wp.tid()
    
    # Get current body data
    q_old = body_q_old[tid]
    qd_old = body_qd_old[tid]
    q_new = body_q_new[tid]  
    qd_new = body_qd_new[tid]
    f = body_f[tid]
    
    # Body properties
    mass = body_mass[tid]
    inertia = body_inertia[tid] 
    
    # Unpack state vectors
    # Old state
    x0 = wp.transform_get_translation(q_old)
    r0 = wp.transform_get_rotation(q_old)
    v0 = wp.spatial_top(qd_old)
    w0 = wp.spatial_bottom(qd_old)
    
    # New state (current guess)
    x1 = wp.transform_get_translation(q_new)
    r1 = wp.transform_get_rotation(q_new)
    v1 = wp.spatial_top(qd_new)
    w1 = wp.spatial_bottom(qd_new)
    
    f_ext = wp.spatial_top(f)   # External force
    t_ext = wp.spatial_bottom(f)  # External torque

    R = wp.quat_to_matrix(r1)
    inertia_world = R @ inertia @ wp.transpose(R)
    
    # Construct 6x6 spatial mass matrix
    # Upper-left 3x3: translational mass matrix
    # Lower-right 3x3: rotational inertia matrix (in world frame)
    # Off-diagonal blocks: zeros (for a body at its center of mass)
    # M = wp.spatial_matrixf(
    #     mass, 0.0, 0.0, 0.0, 0.0, 0.0,
    #     0.0, mass, 0.0, 0.0, 0.0, 0.0,
    #     0.0, 0.0, mass, 0.0, 0.0, 0.0,
    #     0.0, 0.0, 0.0, inertia_world[0, 0], inertia_world[0, 1], inertia_world[0, 2],
    #     0.0, 0.0, 0.0, inertia_world[1, 0], inertia_world[1, 1], inertia_world[1, 2],
    #     0.0, 0.0, 0.0, inertia_world[2, 0], inertia_world[2, 1], inertia_world[2, 2]
    # )
    M = mass * wp.identity(6, dtype=wp.float32)
    M[3:, 3:] = inertia_world

    # gyroscopic and angular damping torques
    t_gyro = -wp.cross(w1,  inertia_world @ w1)
    t_ext += t_gyro
    t_damping = -angular_damping * inertia_world @ w1
    t_ext += t_damping
    
    f_total = wp.spatial_vector(f_ext, t_ext) + contact_f[tid]
    residual[tid] = M @ (qd_new - qd_old) - dt * f_total

    J = M - dt * (contact_jac_vel[tid] + dt* contact_jac_pos[tid])
    
    jacobian[tid] = J
    

@wp.kernel
def solve_newton_step(
    jacobian: wp.array(dtype=wp.spatial_matrixf),
    residual: wp.array(dtype=wp.spatial_vectorf),
    delta_state: wp.array(dtype=wp.spatial_vectorf),
):
    tid = wp.tid()
    
    J = jacobian[tid]
    r = residual[tid]
    delta = solve6x6_gaussian(J, -r)
    delta_state[tid] = delta


@wp.kernel
def update_body_velocity(
    body_qd: wp.array(dtype=wp.spatial_vector),
    delta_state: wp.array(dtype=wp.spatial_vectorf),
    relaxation_factor: float,
    max_velocity: float,
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    qd_current = body_qd[tid]
    delta = delta_state[tid]
    delta_mag = wp.length(delta)
    if delta_mag > max_velocity:
        delta *= max_velocity / delta_mag
    body_qd_out[tid] = qd_current + delta * relaxation_factor

@wp.kernel
def integrate_position(
    body_q_old: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
    dt: float,
    body_q_new: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    
    q_old = body_q_old[tid]
    qd_new = body_qd_new[tid]
    
    # Extract old position and rotation
    x0 = wp.transform_get_translation(q_old)
    r0 = wp.transform_get_rotation(q_old)
    
    v1 = wp.spatial_top(qd_new)
    w1 = wp.spatial_bottom(qd_new)
    
    x1 = x0 + v1 * dt
    
    # Integrate rotation
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)
    
    # Convert back to body origin  
    q_new = wp.transform(x1, r1)
    
    body_q_new[tid] = q_new



@wp.kernel
def compute_residual_norm_kernel(
    delta_state: wp.array(dtype=wp.spatial_vectorf),
    residual: wp.array(dtype=wp.spatial_vectorf),
    residual_norm: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    delta = delta_state[tid]
    r = residual[tid]
    
    delta_norm = wp.norm_l2(delta)
    r_norm = wp.norm_l2(r)
    residual_norm[tid] = r_norm


@wp.func
def compute_forces_and_jacobians(v_rel: wp.vec3, r: wp.vec3,  n: wp.vec3, d: float, ke: float, kd: float, mu: float):

    # decompose relative velocity
    vn = wp.dot(n, v_rel)

    # normal restoring term 
    N = ke * d + kd * vn

    # friction force
    n_n = wp.outer(n, n)
    P = wp.identity(3, dtype=wp.float32) - n_n
    u = P @ v_rel # tangential (slip) velocity

    ft = wp.vec3(0.0)
    vs_sq = wp.length_sq(u) + 1e-4
    vs = wp.sqrt(vs_sq)
    fr = u / vs
    ft = -fr * mu * N

    f = -n * N + ft
    tau = wp.cross(r, f)

    skew_r = wp.skew(r)
    # skew_w = wp.skew(body_w)

    u_u = wp.outer(u, u)

    # force derivatives
    F_x = -ke * n_n
    F_v = -kd * n_n
    # # normal components 

    # friction components
    frn = wp.outer(fr, n)
    F_x += mu * ke * frn
    M = vs_sq * wp.identity(3, dtype=wp.float32) - u_u
    M = M / wp.pow(vs, 3.0)
    F_v += -mu * ( kd * frn + N * M @ P)
 
    # F_w_a = kd * wp.outer(n, n @ skew_r_a)
    # F_w_b = kd * wp.outer(n, n @ skew_r_b)

    # F_R_a = wp.outer(n, ke * n@skew_r_a - kd * n @ (skew_w_a @ skew_r_a))
    # F_R_b = wp.outer(n, ke * n@skew_r_b - kd * n @ (skew_w_b @ skew_r_b))

    # # torque derivatives
    # tau_x_a = skew_r_a @ F_x
    # tau_x_b = skew_r_b @ F_x
    
    # tau_v_a = skew_r_a @ F_v
    # tau_v_b = skew_r_b @ F_v

    # tau_w_a = skew_r_a @ F_w_a
    # tau_w_b = skew_r_b @ F_w_b

    # # tau_R_a = ...
    # # tau_R_b = ...

    # assemble jacobians
    jacob_pos = wp.spatial_matrixf()
    jacob_vel = wp.spatial_matrixf()

    jacob_pos[:3, :3] = F_x
    jacob_vel[:3, :3] = F_v

    f_total = wp.spatial_vector(f, tau)

    return f_total, jacob_pos, jacob_vel

@wp.kernel
def eval_rigid_contacts(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    shape_material_ke: wp.array(dtype=float),
    shape_material_kd: wp.array(dtype=float),
    shape_material_kf: wp.array(dtype=float),
    shape_material_ka: wp.array(dtype=float),
    shape_material_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    # outputs
    contact_force: wp.array(dtype=wp.spatial_vector),
    contact_jac_pos: wp.array(dtype=wp.spatial_matrixf),
    contact_jac_vel: wp.array(dtype=wp.spatial_matrixf),
):
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    # retrieve contact thickness, compute average contact material properties
    ke = 0.0  # contact normal force stiffness
    kd = 0.0  # damping coefficient
    kf = 0.0  # friction force stiffness
    ka = 0.0  # adhesion distance
    mu = 0.0  # friction coefficient
    mass = 0.0
    mat_nonzero = 0
    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]
    if shape_a == shape_b:
        return
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        mat_nonzero += 1
        ke += shape_material_ke[shape_a]
        kd += shape_material_kd[shape_a]
        kf += shape_material_kf[shape_a]
        ka += shape_material_ka[shape_a]
        mu += shape_material_mu[shape_a]
        body_a = shape_body[shape_a]
        mass += body_mass[body_a]
    if shape_b >= 0:
        mat_nonzero += 1
        ke += shape_material_ke[shape_b]
        kd += shape_material_kd[shape_b]
        kf += shape_material_kf[shape_b]
        ka += shape_material_ka[shape_b]
        mu += shape_material_mu[shape_b]
        body_b = shape_body[shape_b]
        mass += body_mass[body_b]
    if mat_nonzero > 0:
        ke /= float(mat_nonzero)
        kd /= float(mat_nonzero)
        kf /= float(mat_nonzero)
        ka /= float(mat_nonzero)
        mu /= float(mat_nonzero)
        mass /= float(mat_nonzero)

    kd *= 2.0 * wp.sqrt(ke * mass)

    # contact normal in world space
    # we invert normal here, such that the n_a pushes a out of b 
    n = -contact_normal[tid] 
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)
    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_com_a = body_com[body_a]
        bx_a = wp.transform_point(X_wb_a, bx_a) + thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, X_com_a)

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_com_b = body_com[body_b]
        bx_b = wp.transform_point(X_wb_b, bx_b) - thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, X_com_b)

    # positive d --> contact is active
    d = wp.dot(n, bx_a - bx_b)
 
    if d < ka:
        return

    # compute contact point velocity
    bv_a = wp.vec3(0.0)
    bv_b = wp.vec3(0.0)
    if body_a >= 0:
        body_v_s_a = body_qd[body_a]
        body_w_a = wp.spatial_bottom(body_v_s_a)
        body_v_a = wp.spatial_top(body_v_s_a)
        bv_a = body_v_a + wp.cross(body_w_a, r_a)

    if body_b >= 0:
        body_v_s_b = body_qd[body_b]
        body_w_b = wp.spatial_bottom(body_v_s_b)
        body_v_b = wp.spatial_top(body_v_s_b)
        bv_b = body_v_b + wp.cross(body_w_b, r_b)

    v_rel = bv_a - bv_b

    if body_a >= 0:
        f_total_a, jacob_pos_a, jacob_vel_a = compute_forces_and_jacobians(v_rel, r_a, n, d, ke, kd, mu)
        wp.atomic_add(contact_force, body_a, f_total_a)
        wp.atomic_add(contact_jac_pos, body_a, jacob_pos_a)
        wp.atomic_add(contact_jac_vel, body_a, jacob_vel_a)
    
    if body_b >= 0:
        f_total_b, jacob_pos_b, jacob_vel_b = compute_forces_and_jacobians(-v_rel, r_b, -n, d, ke, kd, mu)
        wp.atomic_add(contact_force, body_b, f_total_b)
        wp.atomic_add(contact_jac_pos, body_b, jacob_pos_b)
        wp.atomic_add(contact_jac_vel, body_b, jacob_vel_b)






