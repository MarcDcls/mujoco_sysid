import mujoco


class Parameter:
    def __init__(self, value: float, min: float, max: float, optimize: bool = True):
        self.value: float = value
        self.min: float = min
        self.max: float = max


class Actuator:
    def __init__(self, name: str, model: mujoco.MjModel, dof_names: list[str], frictionloss: Parameter, damping: Parameter, armature: Parameter, forcerange: Parameter):
        self.name = name
        self.dof_idx = [model.joint(name).id - 1 for name in dof_names]
        self.frictionloss = frictionloss
        self.damping = damping
        self.armature = armature
        self.forcerange = forcerange

    def get_parameters(self):
        return {
            self.name + "_frictionloss": self.frictionloss,
            self.name + "_damping": self.damping,
            self.name + "_armature": self.armature,
            self.name + "_forcerange": self.forcerange,
        }

    def update_model(self, model: mujoco.MjModel):
        for id in self.dof_idx:
            model.dof_frictionloss[id + 6] = self.frictionloss.value
            model.dof_damping[id + 6] = self.damping.value
            model.dof_armature[id + 6] = self.armature.value
            model.actuator_forcerange[id] = [-self.forcerange.value, self.forcerange.value]


class Body:
    def __init__(
        self,
        name: str,
        model: mujoco.MjModel,
        com_x_offset: Parameter,
        com_y_offset: Parameter,
        com_z_offset: Parameter,
    ):
        self.name = name
        self.body_id = model.body(name).id
        self.com_x_offset = com_x_offset
        self.com_y_offset = com_y_offset
        self.com_z_offset = com_z_offset
        self.initial_com = model.body_ipos[self.body_id].copy()

    def get_parameters(self):
        return {
            self.name + "_com_x_offset": self.com_x_offset,
            self.name + "_com_y_offset": self.com_y_offset,
            self.name + "_com_z_offset": self.com_z_offset,
        }

    def update_model(self, model: mujoco.MjModel):
        model.body_ipos[self.body_id, 0] = self.initial_com[0] + self.com_x_offset.value
        model.body_ipos[self.body_id, 1] = self.initial_com[1] + self.com_y_offset.value
        model.body_ipos[self.body_id, 2] = self.initial_com[2] + self.com_z_offset.value
            

class MujocoModelWrapper:
    """
    MujocoModelWrapper is a class allowing to update friction parameters to do an identification.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuator: list[Actuator],
        body: list[Body] = [],
    ):
        self.model = model
        self.data = data
        self.actuator = actuator
        self.body = body
        
    def get_parameters(self) -> dict[str, Parameter]:
        params: dict[str, Parameter] = {}
        for act in self.actuator:
            params.update(act.get_parameters())
        for body in self.body:
            params.update(body.get_parameters())
        return params
    
    def update_model(self):
        for body in self.body:
            body.update_model(self.model)
        for act in self.actuator:
            act.update_model(self.model)
        mujoco.mj_setConst(self.model, self.data)


if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_path("../humanoid_model/k1/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = 0.005

    arm_actuator = Actuator(
        name="Arm",
        model=model,
        dof_names=["Left_Shoulder_Pitch", "Right_Shoulder_Pitch",
                   "Left_Shoulder_Roll", "Right_Shoulder_Roll", 
                   "Left_Elbow_Pitch", "Right_Elbow_Pitch",
                   "Left_Elbow_Yaw", "Right_Elbow_Yaw"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.001, 0.0, 1.0),
        forcerange=Parameter(10.0, 5.0, 15.0),
    )

    hip_roll_actuator = Actuator(
        name="Hip_Roll",
        model=model,
        dof_names=["Left_Hip_Roll", "Right_Hip_Roll"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0339552, 0.0, 1.0),
        forcerange=Parameter(30.0, 20.0, 45.0),
    )

    hip_pitch_actuator = Actuator(
        name="Hip_Pitch",
        model=model,
        dof_names=["Left_Hip_Pitch", "Right_Hip_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0478125, 0.0, 1.0),
        forcerange=Parameter(25.0, 15.0, 40.0),
    )

    hip_yaw_actuator = Actuator(
        name="Hip_Yaw",
        model=model,
        dof_names=["Left_Hip_Yaw", "Right_Hip_Yaw"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0282528, 0.0, 1.0),
        forcerange=Parameter(20.0, 15.0, 35.0),
    )

    knee_actuator = Actuator(
        name="Knee",
        model=model,
        dof_names=["Left_Knee_Pitch", "Right_Knee_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.095625, 0.0, 1.0),
        forcerange=Parameter(45.0, 30.0, 55.0),
    )

    ankle_roll_actuator = Actuator(
        name="Ankle_Roll",
        model=model,
        dof_names=["Left_Ankle_Roll", "Right_Ankle_Roll"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0565, 0.0, 1.0),
        forcerange=Parameter(20.0, 5.0, 30.0),
    )

    ankle_pitch_actuator = Actuator(
        name="Ankle_Pitch",
        model=model,
        dof_names=["Left_Ankle_Pitch", "Right_Ankle_Pitch"],
        frictionloss=Parameter(0.001, 0.0, 1.0),
        damping=Parameter(0.001, 0.0, 1.0),
        armature=Parameter(0.0565, 0.0, 1.0),
        forcerange=Parameter(20.0, 5.0, 30.0),
    )

    trunk_body = Body(
        name="Trunk",
        model=model,
        com_x_offset=Parameter(0.0, -0.1, 0.1),
        com_y_offset=Parameter(0.0, -0.1, 0.1),
        com_z_offset=Parameter(0.0, -0.1, 0.1),
    )

    wrapper = MujocoModelWrapper(
        model=model, 
        data=data, 
        actuator=
        [
            arm_actuator, 
            hip_roll_actuator, 
            hip_pitch_actuator, 
            hip_yaw_actuator, 
            knee_actuator, 
            ankle_roll_actuator, 
            ankle_pitch_actuator,
        ],
        body=
        [
            trunk_body,
        ],
    )

    print("Initial trunk COM: ", model.body_ipos[model.body("Trunk").id])

    for act in wrapper.actuator:
        for name, param in act.get_parameters().items():
            param.value = 0.1
        act.update_model(model)
    for body in wrapper.body:
        for name, param in body.get_parameters().items():
            param.value = 0.05
        body.update_model(model)
    wrapper.update_model()

    print("Updated trunk COM: ", model.body_ipos[model.body("Trunk").id])

    print("Friction loss: \n", model.dof_frictionloss)
    print("Damping: \n", model.dof_damping)
    print("Armature: \n", model.dof_armature)
    print("Force range: \n", model.actuator_forcerange)