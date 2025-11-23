import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_sphere(body_sphere, radius=0.5)

        # CAPSULE
        body_capsule = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7)

        # CYLINDER (no collision support)
        # body_cylinder = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -4.0, drop_z), q=wp.quat_identity()))
        # builder.add_shape_cylinder(body_cylinder, radius=0.4, half_height=0.6)

        # BOX
        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25)

        # CONE (no collision support)
        # body_cone = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 6.0, drop_z), q=wp.quat_identity()))
        # builder.add_shape_cone(body_cone, radius=0.45, half_height=0.6)

        # MESH (bunny)
        # usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        # usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        # mesh_vertices = np.array(usd_geom.GetPointsAttr().Get())
        # mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        # demo_mesh = newton.Mesh(mesh_vertices, mesh_indices)

        # body_mesh = builder.add_body(
        #     xform=wp.transform(p=wp.vec3(0.0, 4.0, drop_z - 0.5), q=wp.quat(0.5, 0.5, 0.5, 0.5))
        # )
        # builder.add_shape_mesh(body_mesh, mesh=demo_mesh)

        # finalize model
        self.model = builder.finalize()
        angular_damping = 0.05
        self.solver_nr = newton.solvers.SolverNR(self.model, angular_damping=angular_damping)
        self.solver_semi = newton.solvers.SolverSemiImplicit(self.model, angular_damping=angular_damping)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.state_semi_0 = self.model.state()
        self.state_semi_1 = self.model.state()

        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_semi_0)
        self.state_0.body_qd.assign([wp.spatial_vector(0.0, 0.0, 5.0, 0.0, 2.0, 2.0) for _ in range(self.model.body_count)])
        self.state_semi_0.body_qd.assign([wp.spatial_vector(0.0, 0.0, 5.0, 0.0, 2.0, 2.0) for _ in range(self.model.body_count)])

        # Initialize trajectory storage
        self.nr_trajectories = []
        self.semi_trajectories = []
        self.max_trajectory_steps = 1000  # Limit trajectory size
        
        # self.capture()
        self.graph = None

    def extract_state_data(self, state):
        """Extract position and velocity data from a state for trajectory storage."""
        # Convert Warp arrays to numpy arrays for easier access
        body_q_np = state.body_q.numpy()  # transforms
        body_qd_np = state.body_qd.numpy()  # spatial vectors
        
        positions = []
        velocities = []
        
        # Extract body positions and velocities
        for i in range(self.model.body_count):
            # Get body position (3D vector from transform)
            transform = body_q_np[i]
            positions.append([transform[0], transform[1], transform[2]])  # x, y, z from transform
            
            # Get body velocity (6D spatial vector: angular + linear)
            spatial_vel = body_qd_np[i]
            # spatial_vector format: [wx, wy, wz, vx, vy, vz]
            velocities.append([spatial_vel[0], spatial_vel[1], spatial_vel[2], 
                             spatial_vel[3], spatial_vel[4], spatial_vel[5]])
        
        return {
            'positions': np.array(positions),
            'velocities': np.array(velocities)
        }

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_semi_0.clear_forces()

            self.viewer.apply_forces(self.state_0)
            self.viewer.apply_forces(self.state_semi_0)

            self.solver_nr.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.solver_semi.step(self.state_semi_0, self.state_semi_1, self.control, None, self.sim_dt)

            # Store trajectory data (limit storage to avoid memory issues)
            if len(self.nr_trajectories) < self.max_trajectory_steps:
                self.nr_trajectories.append(self.extract_state_data(self.state_1))
                self.semi_trajectories.append(self.extract_state_data(self.state_semi_1))

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_semi_0, self.state_semi_1 = self.state_semi_1, self.state_semi_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        import time
        time.sleep(0.01)

    def compare_trajectories(self, tolerance_pos=1e-3, tolerance_vel=1e-2):
        """Compare trajectories between nr and Semi-implicit solvers."""
        if len(self.nr_trajectories) == 0 or len(self.semi_trajectories) == 0:
            print("Warning: No trajectory data available for comparison")
            return False
        
        if len(self.nr_trajectories) != len(self.semi_trajectories):
            print(f"Warning: Trajectory lengths differ: nr={len(self.nr_trajectories)}, Semi={len(self.semi_trajectories)}")
            return False
        
        print(f"Comparing trajectories with {len(self.nr_trajectories)} timesteps")
        
        # Track maximum differences
        max_pos_diff = 0.0
        max_vel_diff = 0.0
        pos_errors = []
        vel_errors = []
        
        for i in range(len(self.nr_trajectories)):
            nr_state = self.nr_trajectories[i]
            semi_state = self.semi_trajectories[i]
            
            # Compare positions
            pos_diff = np.abs(nr_state['positions'] - semi_state['positions'])
            max_pos_diff_step = np.max(pos_diff)
            max_pos_diff = max(max_pos_diff, max_pos_diff_step)
            pos_errors.append(max_pos_diff_step)
            
            # Compare velocities
            vel_diff = np.abs(nr_state['velocities'] - semi_state['velocities'])
            max_vel_diff_step = np.max(vel_diff)
            max_vel_diff = max(max_vel_diff, max_vel_diff_step)
            vel_errors.append(max_vel_diff_step)
        
        print(f"Maximum position difference: {max_pos_diff:.6f}")
        print(f"Maximum velocity difference: {max_vel_diff:.6f}")
        print(f"Average position error: {np.mean(pos_errors):.6f}")
        print(f"Average velocity error: {np.mean(vel_errors):.6f}")
        
        # Test if trajectories are similar within tolerance
        pos_similar = max_pos_diff < tolerance_pos
        vel_similar = max_vel_diff < tolerance_vel
        
        print(f"Position similarity test (tol={tolerance_pos}): {'PASS' if pos_similar else 'FAIL'}")
        print(f"Velocity similarity test (tol={tolerance_vel}): {'PASS' if vel_similar else 'FAIL'}")
        
        return pos_similar and vel_similar

    def test(self):
        """Test trajectory similarity between solvers."""
        print("Running trajectory similarity test...")
        
        # Run simulation for enough steps to collect data
        if len(self.nr_trajectories) < 100:
            print("Running additional simulation steps to collect trajectory data...")
            for _ in range(100):  # Run 100 simulation frames
                self.simulate()
        
        # Compare the trajectories
        is_similar = self.compare_trajectories()
        
        if is_similar:
            print("SUCCESS: Solver trajectories are similar within tolerance")
        else:
            print("WARNING: Solver trajectories differ significantly")
        
        return is_similar

    def save_trajectories(self, filename_prefix="trajectory"):
        """Save trajectory data to numpy files for later analysis."""
        if len(self.nr_trajectories) == 0 or len(self.semi_trajectories) == 0:
            print("No trajectory data to save")
            return
        
        # Convert trajectory data to arrays
        nr_positions = np.array([state['positions'] for state in self.nr_trajectories])
        nr_velocities = np.array([state['velocities'] for state in self.nr_trajectories])
        
        semi_positions = np.array([state['positions'] for state in self.semi_trajectories])
        semi_velocities = np.array([state['velocities'] for state in self.semi_trajectories])
        
        # Save to files
        np.save(f"{filename_prefix}_nr_positions.npy", nr_positions)
        np.save(f"{filename_prefix}_nr_velocities.npy", nr_velocities)
        np.save(f"{filename_prefix}_semi_positions.npy", semi_positions)
        np.save(f"{filename_prefix}_semi_velocities.npy", semi_velocities)
        
        print(f"Saved trajectory data with {len(self.nr_trajectories)} timesteps to {filename_prefix}_*.npy files")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # self.viewer.log_state(self.state_semi_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    import sys
    
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example
    example = Example(viewer)
    
    # Check if we should run the test
    if "--test" in sys.argv or "test" in sys.argv:
        print("Running trajectory comparison test...")
        result = example.test()
        
        # Optionally save trajectories
        if "--save" in sys.argv or "save" in sys.argv:
            example.save_trajectories("test_free_body")
        
        sys.exit(0 if result else 1)
    else:
        # Run normal simulation with viewer
        newton.examples.run(example, args)
