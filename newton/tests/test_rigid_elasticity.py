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

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_rigid_elasticity_penetration_depth(test, device):
    """Test that rigid contact elasticity produces correct penetration depth at rest.

    At equilibrium, the contact force balances gravity:
        F_contact = k_e * d = F_gravity = m * g
    Therefore: d = m * g / k_e

    Note: The kernel averages the ke of both contacting shapes:
        effective_ke = (ground_ke + sphere_ke) / 2

    This test verifies this relationship for multiple stiffness values.
    """
    mass = 1.0
    gravity = 9.81
    radius = 0.5
    drop_height = 1.0
    spacing = 2.0 
    ground_ke = 1.0e5  # default ke for ground plane

    # Test multiple stiffness values
    stiffness_values = [1.0e2, 1.0e3, 1.0e4, 1.0e5]

    builder = newton.ModelBuilder()

    # Add ground plane
    builder.add_ground_plane()

    # Add all spheres with different stiffness values
    for i, ke in enumerate(stiffness_values):
        kd = ke / 10.0

        shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=ke,
            kd=kd,
            mu=0.5,
        )

        body = builder.add_body(
            xform=wp.transform(wp.vec3(i * spacing, 0.0, drop_height), wp.quat_identity()),
            mass=mass,
            key=f"sphere_{i}",
        )
        builder.add_shape_sphere(body, radius=radius, cfg=shape_cfg)

    model = builder.finalize(device=device)

    # Use XPBD solver with 1 iteration and full relaxation
    solver = newton.solvers.SolverXPBD(
        model,
        iterations=1,
        rigid_contact_relaxation=1.0,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Initialize kinematics
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sim_dt = 1.0 / 2000.0
    settle_time = 2.0  # seconds to settle
    total_steps = int(settle_time / sim_dt)

    # Run simulation
    for _ in range(total_steps):
        state_0.clear_forces()
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    final_body_q = state_0.body_q.numpy()
    final_body_qd = state_0.body_qd.numpy()

    relative_tol = 0.2

    # Verify each sphere
    for i, ke in enumerate(stiffness_values):
        final_z = final_body_q[i, 2]
        effective_ke = (ground_ke + ke) / 2.0
        expected_penetration = mass * gravity / effective_ke
        actual_penetration = radius - final_z

        test.assertAlmostEqual(
            actual_penetration,
            expected_penetration,
            delta=expected_penetration * relative_tol,
            msg=f"ke={ke} (effective={effective_ke}): Expected penetration {expected_penetration:.6f}, got {actual_penetration:.6f}",
        )

        # Verify the object has settled (near-zero velocity)
        linear_vel = np.linalg.norm(final_body_qd[i, :3])
        angular_vel = np.linalg.norm(final_body_qd[i, 3:])

        test.assertLess(
            linear_vel,
            0.01,
            f"ke={ke}: Sphere not at rest (linear velocity = {linear_vel:.6f})",
        )
        test.assertLess(
            angular_vel,
            0.01,
            f"ke={ke}: Sphere not at rest (angular velocity = {angular_vel:.6f})",
        )


devices = get_test_devices()


class TestRigidElasticity(unittest.TestCase):
    pass


add_function_test(
    TestRigidElasticity,
    "test_rigid_elasticity_penetration_depth",
    test_rigid_elasticity_penetration_depth,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)

