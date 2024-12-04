class FuncInfo(object):
    def __init__(self, funcname: str, funcdescription: str, arg_and_description_list: dict[str, str], example_str: str):
        """
        Function info object

        :param funcname: The name of the function to be represented
        :param funcdescription: Description of the function
        :param arg_and_description_list: args and descriptions like {"arg1": "arg1 description", ...}
        :param example_str: Example function call with a description
        """
        self.name = funcname
        self.desc = funcdescription
        self.args = arg_and_description_list
        self.example = example_str

    def __str__(self):
        return f"Function '{self.name}'\n\tArgs:\n\t\t{self.args}\n\tDesc:\n\t\t{self.desc}\n\tExample:\n\t\t{self.example}"
