from btp.bt.base_nodes import Action
from btp.bt import Status


class GoTo(Action):
    can_be_expanded = True

    def __init__(self, agent_name, target_obj):
        super().__init__(*agent_name, target_obj)
        self.agent_name = agent_name
        self.target_obj = target_obj

        self.act_max_step = 30
        self.act_cur_step = 0

    def update(self) -> Status:
        # script = [f'<char0> [{self.__class__.__name__.lower()}] <{self.args[0].lower()}> (1)']

        if self.num_args==1:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1)']
        else:
            script = [f'<char0> [{self.action_class_name.lower()}] <{self.args[0].lower()}> (1) <{self.args[1].lower()}> (1)']


        self.env.run_script(script,verbose=True,camera_mode="PERSON_FROM_BACK") # FIRST_PERSON
        print("script: ",script)
        self.change_condition_set()

        return Status.RUNNING