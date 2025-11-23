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

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    compute_residual_and_jacobian,
    solve_newton_step,
    update_body_velocity,
    add_gravity_kernel,
    add_external_wrenches_kernel,
    integrate_position,
    compute_residual_norm_kernel,
    eval_rigid_contacts,
)



class SolverNR(SolverBase):
    """
    A maximal coordinate, velocity-level, block-jacobi implicit solver using Newton-Raphson iterations.
    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        friction_smoothing: float = 1.0,
        max_newton_iterations: int = 10,
        newton_tolerance: float = 1e-4,
        relaxation_factor: float = 0.9,
        enable_tri_contact: bool = True,
        predictor: int = 0,
        verbose: bool = False,
    ):
        
        super().__init__(model=model)
        self.angular_damping = angular_damping
        self.friction_smoothing = friction_smoothing
        self.max_newton_iterations = max_newton_iterations
        self.newton_tolerance = newton_tolerance
        self.enable_tri_contact = enable_tri_contact
        self.relaxation_factor = relaxation_factor
        self.predictor = predictor
        self.verbose = verbose

        self.allocate_model_aux_vars(model)

    def allocate_model_aux_vars(self, model: Model):
        body_count = model.body_count
        
        self.jacobian_matrices = wp.zeros((body_count,), dtype=wp.spatial_matrixf, device=model.device)
        self.residual_vectors = wp.zeros((body_count,), dtype=wp.spatial_vectorf, device=model.device)
        self.delta_state = wp.zeros((body_count,), dtype=wp.spatial_vectorf, device=model.device)
        self.temp_body_f = wp.zeros((body_count,), dtype=wp.spatial_vectorf, device=model.device)
        self.contact_jac_pos = wp.zeros((body_count,), dtype=wp.spatial_matrixf, device=model.device)
        self.contact_jac_vel = wp.zeros((body_count,), dtype=wp.spatial_matrixf, device=model.device)
        self.residual_norm = wp.zeros((body_count,), dtype=wp.float32, device=model.device) # per body residual norm
        self.contact_f = wp.zeros((body_count,), dtype=wp.spatial_vectorf, device=model.device)


    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        with wp.ScopedTimer("simulate", False):
            if state_in.particle_count:
                raise NotImplementedError("Particle forces not implemented for NR solver")

            if not state_in.body_count:
                return state_out

            model = self.model

            if control is None:
                control = model.control(clone_variables=False)
            
            # Perform Newton-Raphson iterations
            self._solve_implicit_system(
                model, state_in, state_out, contacts, dt
            )

            return state_out

    def _set_initial_state(self, model: Model, state_in: State, state_out: State, dt: float):
        # set initial guess for state
        if self.predictor == 0:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)
        elif self.predictor == 1:
            # use semi-implicit integration step as initial guess
            self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

    def _solve_implicit_system(
        self,
        model: Model,
        state_in: State,
        state_out: State,
        contacts: Contacts,
        dt: float,
    ):
        """Solve the implicit system using block-jacobi Newton-Raphson iterations."""
        # TODO: We ignored COM for now

        relaxation_factor = 1.0
        self._set_initial_state(model, state_in, state_out, dt)

        self.temp_body_f.zero_()
        self._add_gravity_forces(model, self.temp_body_f)
        self._add_external_wrenches(model, state_in.body_f, self.temp_body_f)

        for iteration in range(self.max_newton_iterations):
            self.contact_jac_pos.zero_()
            self.contact_jac_vel.zero_()
            self.contact_f.zero_()

            self._integrate_positions(model, state_in, state_out, dt)
            if contacts is not None:
                self._eval_body_contacts(model, state_out, contacts)

            # Compute residual r and jacobian J for each body
            wp.launch(
                kernel=compute_residual_and_jacobian,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_out.body_q,
                    state_out.body_qd,
                    self.temp_body_f,
                    model.body_mass,
                    model.body_inertia,
                    self.contact_f,
                    self.contact_jac_pos,
                    self.contact_jac_vel,
                    dt,
                    self.angular_damping,
                ],
                outputs=[self.residual_vectors, self.jacobian_matrices],
                device=model.device,
            )
            
            # Solve linear systems for each body: J * delta_v = -r
            wp.launch(
                kernel=solve_newton_step,
                dim=model.body_count,
                inputs=[self.jacobian_matrices, self.residual_vectors],
                outputs=[self.delta_state],
                device=model.device,
            )

            # Update velocities: v += delta_v
            wp.launch(
                kernel=update_body_velocity,
                dim=model.body_count,
                inputs=[
                    state_out.body_qd,
                    self.delta_state,
                    relaxation_factor,
                    10000.0,
                ],
                outputs=[state_out.body_qd],
                device=model.device,
            )

            wp.launch(
                kernel=compute_residual_norm_kernel,
                dim=model.body_count,
                inputs=[self.delta_state, self.residual_vectors],
                outputs=[self.residual_norm],
                device=model.device,
            )
            relaxation_factor *= self.relaxation_factor
            if self.residual_norm.numpy().mean() < self.newton_tolerance:
                break

        print(f"{iteration}, final norm: {self.residual_norm.numpy().mean()}")
        print(relaxation_factor)
        final_norm = self.residual_norm.numpy()
        #if (final_norm > self.newton_tolerance).any():
        #     print("Newton iterations did not converge, final residual: ", final_norm)

        # After Newton iterations converge, integrate positions using final velocities
        self._integrate_positions(model, state_in, state_out, dt)

    def _eval_body_contacts(self, model: Model, state: State, contacts: Contacts):
        wp.launch(
            kernel=eval_rigid_contacts,
            dim=contacts.rigid_contact_max,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.body_mass,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_kf,
                model.shape_material_ka,
                model.shape_material_mu,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_thickness0,
                contacts.rigid_contact_thickness1,
                # self.friction_smoothing,
            ],
            outputs=[self.contact_f, self.contact_jac_pos, self.contact_jac_vel],
            device=model.device,
        )
    def _add_gravity_forces(self, model: Model, body_f: wp.array):
        wp.launch(
            kernel=add_gravity_kernel,
            dim=model.body_count,
            inputs=[model.body_mass, model.gravity],
            outputs=[body_f],
            device=model.device,
        )
    
    def _add_external_wrenches(self, model: Model, external_wrenches: wp.array, body_f: wp.array):

        wp.launch(
            kernel=add_external_wrenches_kernel,
            dim=model.body_count,
            inputs=[external_wrenches],
            outputs=[body_f],
            device=model.device,
        )
        
    def _integrate_positions(self, model: Model, state_in: State, state_out: State, dt: float):

        wp.launch(
            kernel=integrate_position,
            dim=model.body_count,
            inputs=[
                state_in.body_q,           # Initial positions
                state_out.body_qd,         # Final velocities from Newton iterations
                dt,
            ],
            outputs=[state_out.body_q],
            device=model.device,
        )