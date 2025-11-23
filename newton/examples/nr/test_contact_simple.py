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
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        ke = 1e6  
        kd = 0.5
        kf = 1e4
        mu = 0.5
        
        # Ground plane with material configuration
        shape_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=kd, kf=kf, mu=mu)
        builder.add_ground_plane(cfg=shape_cfg)

        # z height to drop shapes from
        drop_z = 2.0

        # SPHERE
        body_sphere = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, -2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_sphere(body_sphere, radius=0.5, cfg=shape_cfg)

        # CAPSULE
        body_capsule = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_capsule(body_capsule, radius=0.3, half_height=0.7, cfg=shape_cfg)

        # BOX
        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.5, hy=0.35, hz=0.25, cfg=shape_cfg)

        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z + 1), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.4, hy=0.3, hz=0.25, cfg=shape_cfg)

        body_box = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 2.0, drop_z + 2), q=wp.quat_identity()))
        builder.add_shape_box(body_box, hx=0.3, hy=0.25, hz=0.25, cfg=shape_cfg)

        # finalize model
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverNR(self.model)
        # self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_0.body_qd.assign([wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(self.model.body_count)])

        # self.capture()
        self.graph = None


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

            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        import time
        time.sleep(0.01)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example
    example = Example(viewer)
    
    # Run normal simulation with viewer
    newton.examples.run(example, args)
